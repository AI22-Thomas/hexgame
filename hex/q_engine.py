# imports for the neural network
import os
import time

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

from hex.adversary.base_adversary import BaseAdversary
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
    def __init__(self, env: HexEnv, model: QModel, adversary: BaseAdversary, memory_length=1000, cpu=False, chart=True):
        self.model = model
        self.env = env
        self.chart = chart
        if cpu:
            self.device = "cpu"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda")

        self.memory = ReplayMemory(length=memory_length)
        # number of actions in gym environment
        self.n_actions = len(self.env.action_space())
        # dimensionality of state observations in gym environment
        state, _ = self.env.reset()
        self.reward_history = []
        self.train_reward_history = []
        self.model.initialize_networks(self.device)
        self.adversary = adversary
        self.adversary.init(self)

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
                # mask the actions that are not in the action set (should not be played)
                mask = [-2] * len(net_values[0])
                for i in action_set:
                    mask[i] = 0
                mask = torch.tensor(mask, device=self.device, dtype=torch.float32)
                net_values = net_values + mask
                return net_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([random.sample(action_set, 1)], device=self.device,
                                dtype=torch.long)

    def play(self, env, games=10, play_as_black=False, randomColorOff=False, printBoard=False,
             playWithRandomStart=False):

        rewards = []
        alreadyDoneStartMovesW = []
        alreadyDoneStartMovesB = []

        if (playWithRandomStart):
            totalActionSize = env.board_size * env.board_size
        else:
            totalActionSize = games

        # for i in range(env.board_size * env.board_size):
        for i in range(totalActionSize):
            # initialize the environment and get the state
            state, _ = env.reset()
            # coerce the state to torch tensor type
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            if randomColorOff == False:
                play_as_black = random.random() > 0.5

            # Current: always a random action at start
            # TODO: should be a valid move that has not been done before

            # select action 1 
            if play_as_black:
                action = self.adversary.get_action(state, self)
            else:
                if (playWithRandomStart):
                    action = None
                    while (action == None):
                        # TODO IMPROVE TO TAKE ONE ACTION AFTER EACH OTHER AND NOT SEARCH RANDOMLY
                        action = self._eps_greedy_action(state, eps=2)
                        if (action in alreadyDoneStartMovesW):
                            action = None
                        else:
                            alreadyDoneStartMovesW.append(action)
                else:
                    action = self._eps_greedy_action(state, eps=0)

            # do the action
            observation, _, _, _ = env.step(action.item())
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

            # select action 2 
            if play_as_black:
                if (playWithRandomStart):
                    action = None
                    while (action == None):
                        # TODO IMPROVE TO TAKE ONE ACTION AFTER EACH OTHER AND NOT SEARCH RANDOMLY
                        action = self._eps_greedy_action(state, eps=2)
                        if (i == totalActionSize - 1):
                            break
                        if (action in alreadyDoneStartMovesB):
                            action = None
                        else:
                            alreadyDoneStartMovesB.append(action)
                else:
                    action = self._eps_greedy_action(state, eps=0)
            else:
                action = self.adversary.get_action(state, self)

            # do the action
            state, _, _, _ = self.env.step(action.item())
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if printBoard and i & 10:
                env.engine.print()

            # Go into Play
            for t in count():
                if play_as_black:
                    action = self.adversary.get_action(state, self)
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
                    action = self.adversary.get_action(observation, self)

                observation2, reward2, terminated2, _ = env.step(action.item())
                if play_as_black:
                    reward2 = -reward2

                if terminated2:
                    rewards.append(reward2)
                    break

                state = torch.tensor(observation2, dtype=torch.float32, device=self.device).unsqueeze(0)
        return rewards

    def learn(self, num_episodes=500,
              batch_size=100,
              gamma=0.99,
              eps_start=0.9,
              eps_end=0.01,
              eps_decay=1000,
              soft_update=True,
              target_net_update_rate=0.005,
              learning_rate=1e-4,
              eval_every=10,
              save_every=100,
              start_from_model=None,
              random_start=False,
              save_path="models/model.pt",
              evaluate_runs=100,
              clip_grads=100,
              # 0 = white, 1 = black, 0.5 = random
              playAsColor=0.5):

        if start_from_model is not None:
            if os.path.isfile(start_from_model):
                self.model.load_model(start_from_model)

        def eps_by_step(step):
            return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)

        #optimizer = optim.SGD(self.model.policy_net.parameters(), lr=learning_rate, momentum=0.9)
        #adam optimizer
        #optimizer = optim.Adam(self.model.policy_net.parameters(), lr=learning_rate)
        #other optimizer 
        #optimizer = optim.RMSprop(self.model.policy_net.parameters(), lr=learning_rate)
        #other optimizer 
        #optimizer = optim.Adadelta(self.model.policy_net.parameters(), lr=learning_rate)
        #best optimizer for 7x7 hex game
        optimizer = optim.Adagrad(self.model.policy_net.parameters(), lr=learning_rate)
        steps_done = 0

        winners = []
        for i_episode in range(num_episodes):
            # May update the environment, so do that before resetting
            self.adversary.update(self, epoch=i_episode, random_start=random_start, showPlot=True)

            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            play_as_white = random.random() > playAsColor

            if not play_as_white:
                # Ensure one move is played, so the agent is playing black
                action = self.adversary.get_action(state, self)
                state, _, _, _ = self.env.step(action.item())
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                first_state = state
            # random starting move
            if random_start:
                action = torch.tensor([random.sample(self.env.action_space(), 1)], device=self.device,
                                      dtype=torch.long)
                state, reward, terminated, next_action_space = self.env.step(action.item())
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                #if not play_as_white:
                #    self.memory.save(first_state, action, state, reward, next_action_space)
                #    self.optimize_model(optimizer, batch_size, gamma, clip_grads=clip_grads)


                action = self.adversary.get_action(state, self)
                state, _, _, _ = self.env.step(action.item())
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in count():
                steps_done += 1

                action = self._eps_greedy_action(state, eps=eps_by_step(steps_done))
                observation, reward, terminated, next_action_space = self.env.step(action.item())

                taken_action = action
                reward_t = torch.tensor([reward if play_as_white else -reward], device=self.device)
                done = terminated
                observation2 = observation
                next_action_space2 = None
                if not terminated:
                    action = self.adversary.get_action(
                        torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0), self)
                    observation2, reward2, terminated2, next_action_space2 = self.env.step(action.item())

                    if terminated2:
                        reward_t = torch.tensor([reward2 if play_as_white else -reward2], device=self.device)
                        done = terminated2

                if not done:
                    next_state = torch.tensor(observation2, dtype=torch.float32, device=self.device).unsqueeze(0)
                    # create mask from next action space (actions that are not in the action space should not be played)
                    mask = [-2] * self.n_actions
                    for i in next_action_space2:
                        mask[i] = 0
                    next_action_space = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    next_state = None
                    next_action_space = None

                self.memory.save(state, taken_action, next_state, reward_t, next_action_space)

                state = next_state

                self.optimize_model(optimizer, batch_size, gamma, clip_grads=clip_grads)

                # soft update
                if soft_update:
                    for target_param, param in zip(self.model.target_net.parameters(),
                                                   self.model.policy_net.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - target_net_update_rate) +
                                                param.data * target_net_update_rate)

                else:
                    if steps_done % target_net_update_rate == 0:
                        self.model.target_net.load_state_dict(self.model.policy_net.state_dict())

                if done:
                    self.train_reward_history.append(reward_t.item())
                    break

            # Save if model or adv. won
            if (play_as_white):
                if (self.env.engine.winner == 1):
                    # print("Won as Black")
                    winners.append(1)
                else:
                    winners.append(-1)
            else:
                if (self.env.engine.winner == -1):
                    # print("Won as Black")
                    winners.append(1)
                else:
                    winners.append(-1)
            if i_episode % (eval_every / 4) == 0:
                print("Self Wins: ", winners.count(1), "Adv Wins: ", winners.count(-1))

            if i_episode % (eval_every) == 0:
                winners.clear()
                # self.evaluate(title="Episode {} finished after {} timesteps".format(i_episode, t + 1),
                #          runs=evaluate_runs, clear=True)
            if i_episode % save_every == 0:
                torch.save(self.model.policy_net.state_dict(), save_path)

    def evaluate(self, runs=100, title="", clear=False):
        """
        Plot the reward history to standard output.
        """
        rewards = self.play(self.env, runs, playWithRandomStart=True)
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

    def optimize_model(self, optimizer, batch_size, gamma, clip_grads):
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
            values = torch.add(values, next_action_spaces_mask)
            next_state_values[non_final_mask] = values.max(1)[0].detach()
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Mean Squared Error loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the policy network
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), clip_grads)
        optimizer.step()
