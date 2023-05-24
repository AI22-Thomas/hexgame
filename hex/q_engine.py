# imports for the neural network
import os

import torch
import torch.nn as nn
import torch.optim as optim

# other imports
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque, OrderedDict
from itertools import count
from IPython.display import clear_output

from hex.hex_env import HexEnv
from hex.qmodels.q_model import QModel

TransitionData = namedtuple('Transition', ['state', 'action', 'next_state', 'reward', 'next_action_space'])


class ReplayMemory(object):
    def __init__(self, length: int):
        # the deque class is designed for popping from the right and from the left
        self.memory = deque([], maxlen=length)

    def save(self, state, action, next_state, reward, next_action_space):
        self.memory.append(TransitionData(state, action, next_state, reward, next_action_space))

    def sample(self, batch_size: int):
        """
        Bootstrap 'batch_size' transitions from the memory.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Get the number of transitions in memory.
        """
        return len(self.memory)


class QEngine(object):
    def __init__(self, env: HexEnv, model: QModel, memory_length=1000, cpu=False, clip_grads=100, chart=True):
        self.model = model
        self.clip_grads = clip_grads
        self.env = env
        self.chart = chart
        if cpu:
            self.device = "cpu"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(length=memory_length)
        # number of actions in gym environment
        self.n_actions = len(self.env.action_space())
        # dimensionality of state observations in gym environment
        state, _ = self.env.reset()
        self.reward_history = []
        self.train_reward_history = []
        self.model.initialize_networks(self.device)
        self.use_adversary = False

    def _eps_greedy_action(self, state, eps, action_set=None, net=None):
        if action_set is None:
            action_set = self.env.action_space()
        if net is None:
            net = self.model.policy_net
        """
        Returns an 'eps'-greedy action.
        Does not modify the object.
        """
        if random.random() > eps:
            # deactivating grad computation in torch makes it a little faster
            with torch.no_grad():
                # t.max(1) returns the largest column value of each row
                # the second column of the result is the index of the maximal element
                net_values = net(state)
                if len(net_values[0]) == 1:
                    print("wtf")
                # mask the actions that are not in the action set (should not be played)
                mask = [-2] * len(net_values[0])
                for i in action_set:
                    mask[i] = 1
                mask = torch.tensor(mask, device=self.device, dtype=torch.float32)
                net_values = net_values * mask
                return net_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([random.sample(action_set, 1)], device=self.device,
                                dtype=torch.long)

    def play(self, env, games=10, adversary=False):

        rewards = []
        for i in range(games):
            # initialize the environment and get the state
            state, _ = env.reset()
            # coerce the state to torch tensor type
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            play_as_black = random.random() > 0.5
            for t in count():
                if play_as_black:
                    if adversary:
                        action = self._eps_greedy_action(state, eps=0, net=self.model.adv_net)
                    else:
                        action = self._eps_greedy_action(state, eps=2)
                else:
                    action = self._eps_greedy_action(state, eps=0)
                # select action
                observation, reward, terminated, next_actions = env.step(action.item())
                if play_as_black:
                    reward = -reward

                if terminated:
                    rewards.append(reward)
                    break
                observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                if play_as_black:
                    action = self._eps_greedy_action(observation, eps=0)
                else:
                    if adversary:
                        action = self._eps_greedy_action(observation, eps=0, net=self.model.adv_net)
                    else:
                        action = self._eps_greedy_action(observation, eps=2)

                observation2, reward2, terminated2, _ = env.step(action.item())
                if play_as_black:
                    reward2 = -reward2

                if terminated2:
                    rewards.append(reward2)
                    break

                state = torch.tensor(observation2, dtype=torch.float32, device=self.device).unsqueeze(0)
        return rewards

    def adversary_move(self, state):
        if self.use_adversary:
            return self._eps_greedy_action(
                state,
                eps=0,
                net=self.model.adv_net)
        else:
            return torch.tensor([random.sample(self.env.action_space(), 1)], device=self.device,
                                dtype=torch.long)

    def learn(self, num_episodes=500,
              batch_size=100,
              gamma=0.99,
              eps_start=0.9,
              eps_end=0.01,
              eps_decay=1000,
              target_net_update_rate=0.005,
              learning_rate=1e-4,
              eval_every=10,
              save_every=100,
              start_from_model=None,
              random_start=False,
              save_path="models/model.pt",
              evaluate_runs=100,
              adversary_threshold=0.7,
              self_play=False,
              soft_update=True):

        if start_from_model is not None:
            if os.path.isfile(start_from_model):
                self.model.load_model(start_from_model)

        self.use_adversary = self_play
        self.model.update_adv_net()

        def eps_by_step(step):
            return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)

        optimizer = optim.SGD(self.model.policy_net.parameters(), lr=learning_rate, momentum=0.9)

        steps_done = 0

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            play_as_white = random.random() > 0.5
            # random starting move
            if random_start:
                if play_as_white:
                    action = torch.tensor([random.sample(self.env.action_space(), 1)], device=self.device,
                                          dtype=torch.long)
                else:
                    action = self.adversary_move(state)
                state, _, _, _ = self.env.step(action.item())
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                if play_as_white:
                    action = self.adversary_move(state)
                else:
                    action = torch.tensor([random.sample(self.env.action_space(), 1)], device=self.device,
                                          dtype=torch.long)
                state, _, _, _ = self.env.step(action.item())
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                steps_done += 1
                def play_white():
                    action = self._eps_greedy_action(state, eps=eps_by_step(steps_done))
                    observation, reward, terminated, next_action_space = self.env.step(action.item())

                    taken_action = action
                    reward_t = torch.tensor([reward], device=self.device)
                    done = terminated
                    observation2 = observation
                    next_action_space2 = None
                    if not terminated:
                        action = self.adversary_move(
                            torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                        )
                        observation2, reward2, terminated2, next_action_space2 = self.env.step(action.item())

                        if terminated2:
                            reward_t = torch.tensor([reward2], device=self.device)
                            done = terminated2
                    return taken_action, observation2, reward_t, done, next_action_space2

                def play_black():
                    action = self.adversary_move(state)

                    observation, _, terminated, next_action_space = self.env.simulate(action.item())
                    if terminated:
                        return play_white()
                    self.env.step(action.item())

                    action = self._eps_greedy_action(
                        torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0),
                        eps=eps_by_step(steps_done),
                        action_set=next_action_space)
                    observation, reward, terminated, next_action_space = self.env.step(action.item())
                    taken_action = action
                    # flip reward as we are playing as black
                    reward_t = torch.tensor([-reward], device=self.device)
                    done = terminated
                    observation2 = observation
                    next_action_space2 = None
                    if terminated:
                        done = terminated
                    else:
                        # simulate adversary move
                        action = self.adversary_move(
                            torch.tensor(observation2, dtype=torch.float32, device=self.device).unsqueeze(0))
                        observation2, reward2, terminated2, next_action_space2 = self.env.simulate(action.item())
                        if terminated2:
                            # flip reward as we are playing as black
                            reward_t = torch.tensor([-reward2], device=self.device)
                            done = terminated2

                    return taken_action, observation2, reward_t, done, next_action_space2

                action, next_state, reward_t, done, next_action_space = play_white() if play_as_white else play_black()

                if not done:
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    # create mask from next action space (actions that are not in the action space should not be played)
                    mask = [-2] * self.n_actions
                    for i in next_action_space:
                        mask[i] = 1
                    next_action_space = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    next_state = None
                    next_action_space = None

                self.memory.save(state, action, next_state, reward_t, next_action_space)

                state = next_state

                self.optimize_model(optimizer, batch_size, gamma)

                # soft update
                if soft_update:
                    target_net_state_dict = self.model.target_net.state_dict()
                    policy_net_state_dict = self.model.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * target_net_update_rate + \
                                                     target_net_state_dict[key] * (1.0 - target_net_update_rate)
                    self.model.target_net.load_state_dict(target_net_state_dict)
                else:
                    if steps_done % target_net_update_rate == 0:
                        self.model.target_net.load_state_dict(self.model.policy_net.state_dict())

                if done:
                    self.train_reward_history.append(reward_t.item())
                    break
            if i_episode % eval_every == 0:
                rew = self.evaluate(title="Episode {} finished after {} timesteps".format(i_episode, t + 1),
                                    runs=evaluate_runs, clear=True)
                if rew > adversary_threshold:
                    # update adversary (copy policy net to adversary)
                    self.model.update_adv_net()
            if i_episode % save_every == 0:
                torch.save(self.model.policy_net.state_dict(), save_path)

    def evaluate(self, runs=100, title="", clear=False):
        """
        Plot the reward history to standard output.
        """
        rewards = self.play(self.env, runs, adversary=self.use_adversary)
        avg_rew = sum(rewards) / len(rewards)
        self.reward_history.append(avg_rew)

        if clear:
            clear_output(wait=True)
        print(title)
        print("Average reward: {}".format(sum(rewards) / len(rewards)))

        if self.chart:
            plt.figure(figsize=(10, 5))
            plt.title(title)
            plt.plot(self.reward_history)
            plt.show()
        return avg_rew

    def optimize_model(self, optimizer, batch_size, gamma):
        """
        Performs one step of the optimization (on the policy network).
        """

        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = TransitionData(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        next_action_spaces_mask = torch.cat([s for s in batch.next_action_space if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            values = self.model.target_net(non_final_next_states)
            # mask values that should not be considered
            values = torch.mul(values, next_action_spaces_mask)
            next_state_values[non_final_mask] = values.max(1)[0].detach()
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Mean Squared Error loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the policy network
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), self.clip_grads)
        optimizer.step()
