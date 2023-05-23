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

# convenience class to keep transition data straight
# is used inside 'replayMemory'
TransitionData = namedtuple('Transition', ['state', 'action', 'next_state', 'reward', 'next_action_space'])


class ReplayMemory(object):
    """
    Store transitions consisting of 'state', 'action', 'next_state', 'reward'.

    Attributes
    ----------
    memory : collections.deque
        Here we store named tuples of the class 'transitionData'.
        A deque is a data structure that allows appending/popping "from the right" and "from the left":
        https://en.wikipedia.org/wiki/Double-ended_queue
    """

    def __init__(self, length: int):
        # the deque class is designed for popping from the right and from the left
        self.memory = deque([], maxlen=length)

    def save(self, state, action, next_state, reward, next_action_space):
        """
        Save the transition consisting of 'state', 'action', 'next_state', 'reward'.
        """
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


# Q-NETWORK
# def make_torch_net(input_length: int, output_length: int):
#     layers = []
#     layer_num = 0
#     layers.append((str(layer_num), nn.Linear(input_length, 128)))
#     layer_num += 1
#     layers.append((str(layer_num), nn.ReLU()))
#     layer_num += 1
#     # for i in range(2):
#     #     layers.append((str(layer_num), nn.Linear(128, 128)))
#     #     layer_num += 1
#     #     layers.append((str(layer_num), nn.ReLU()))
#     #     layer_num += 1
#     layers.append((str(layer_num), nn.Linear(128, output_length)))
#     net = nn.Sequential(OrderedDict(layers))
#     print(net)
#     return net


def make_torch_net(input_length: int, output_length: int):
    net = nn.Sequential()
    conv = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
    # 3x3 weights with 6 channels
    weights = torch.zeros(32, 6, 3, 3)
    # set weights for all channels to this:
    # 0 1 1
    # 1 1 1
    # 1 1 0
    for i in range(6):
        weights[:, i, :, :] = torch.tensor([[0, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 0]])
    with torch.no_grad():
        conv.weight = nn.Parameter(weights)

    net.add_module("conv", conv)
    net.add_module("relu1", nn.ReLU())
    net.add_module("flatten", nn.Flatten())
    net.add_module("fc1", nn.Linear(32 * input_length * input_length, 128))
    net.add_module("relu2", nn.ReLU())
    net.add_module("fc2", nn.Linear(128, output_length))
    net.add_module("sigmoid", nn.Sigmoid())
    print(net)
    return net


# MAIN WRAPPER CLASS
class DeepQ(object):
    """
    Deep Q Learning wrapper.

    Attributes
    ----------
    env : gymnasium.env
        Defaults to the "CartPole-v1" environment.
    device : cuda.device
        The hardware used by torch in computation
    memory : ReplayMemory
        The transition memory of the q-learner.
    n_actions : int
        Number of actions in the environment.
    n_observations : int
        Dimensionality of the state vector of the environment.
    """

    def __init__(self, env, memory_length=1000):
        self.env = env
        self.device = "cpu"
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(length=memory_length)
        # number of actions in gym environment
        self.n_actions = len(self.env.action_space())
        # dimensionality of state observations in gym environment
        state, _ = self.env.reset()
        self.input_dim = state.shape[1]
        # self.input_dim = len(state)
        self.reward_history = []

    def initialize_networks(self):
        """
        Sets up policy and target networks with 'hidden' hidden layers of given 'width'.
        Activations are ReLU.
        The dimensionalities of input and output layer are chosen automatically.
        """
        # set up policy net
        self.policy_net = make_torch_net(input_length=self.input_dim, output_length=self.n_actions).to(self.device)
        # set up target net
        self.target_net = make_torch_net(input_length=self.input_dim, output_length=self.n_actions).to(self.device)
        # copy parameters of policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _eps_greedy_action(self, state, eps, action_set=None):

        if action_set is None:
            action_set = self.env.action_space()
        """
        Returns an 'eps'-greedy action.
        Does not modify the object.
        """
        if random.random() > eps:
            # deactivating grad computation in torch makes it a little faster
            with torch.no_grad():
                # t.max(1) returns the largest column value of each row
                # the second column of the result is the index of the maximal element
                net_values = self.policy_net(state)
                mask = [-2] * len(net_values[0])
                for i in action_set:
                    mask[i] = 1
                mask = torch.tensor(mask, device=self.device, dtype=torch.float32)
                net_values = net_values + mask
                return net_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([random.sample(action_set, 1)], device=self.device,
                                dtype=torch.long)

    def play(self, env, num_steps=500):
        """
        Play 'num_steps' using the current policy network.
        During play the environment is rendered graphically.
        """
        # initialize the environment and get the state
        state, _ = env.reset()
        # coerce the state to torch tensor type
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        for i in range(num_steps):
            action = self._eps_greedy_action(state, eps=0)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            if done:
                break
            else:
                state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        env.close()

    def learn(self, num_episodes=500, batch_size=100, gamma=0.99, eps_start=0.9, eps_end=0.01, eps_decay=1000,
              target_net_update_rate=0.005, learning_rate=1e-4, print_every=10, save_every=100):
        """
        Train using the deep q-learning algorithm.

        Parameters
        ----------
        num_episodes : int
            Number of rounds to be played.
            Note that a round lasts longer if the model is good.
            Hence the time a round takes increases during training.
        batch_size : int
            Size of the dataset sampled from the memory for a gradient step.
        gamma : float
            Discount rate.
        eps_start : float
            Epsilon for epsilon-greedy action selection at the start of training.
        eps_end : float
            Epsilon for epsilon-greedy action selection at the end of training.
        eps_decay : float
            Decay rate of epsilon for epsilon-greedy action selection during training.
        target_net_update_rate : float
            Should be between 0 and 1.
            Determines the mix-in rate of the parameters of the policy network into the target network.
            This ensures that the target network lags behind enough to stabilize the learning task.
        learning_rate : float
            Learning rate for the adam optimizer of torch.
        """

        def eps_by_step(step):
            return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)

        optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        steps_done = 0

        best_avg_reward = -1

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            # # random starting move
            # action = torch.tensor([random.sample(self.env.action_space(), 1)], device=self.device,
            #                       dtype=torch.long)
            # _, _, _, _ = self.env.step(action.item())
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                steps_done += 1
                action = self._eps_greedy_action(state, eps=eps_by_step(steps_done))
                observation, reward, terminated, next_action_space = self.env.step(action.item())

                reward_t = torch.tensor([reward], device=self.device)
                done = terminated
                observation2 = observation
                if not terminated:
                    action = torch.tensor([random.sample(self.env.action_space(), 1)], device=self.device,
                                          dtype=torch.long)
                    observation2, reward2, terminated2, next_action_space2 = self.env.step(action.item())

                    if terminated2:
                        reward_t = torch.tensor([reward2], device=self.device)
                        done = terminated2

                if not done:
                    next_state = torch.tensor(observation2, dtype=torch.float32, device=self.device).unsqueeze(0)
                    # create mask from next action space
                    mask = [-2] * self.n_actions
                    for i in next_action_space2:
                        mask[i] = 1
                    next_action_space = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    next_state = None
                    next_action_space = None

                self.memory.save(state, action, next_state, reward_t, next_action_space)

                state = next_state

                self.optimize_model(optimizer, batch_size, gamma)

                # soft update
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * target_net_update_rate + \
                                                 target_net_state_dict[key] * (1.0 - target_net_update_rate)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.reward_history.append(reward)
                    break
            if i_episode % print_every == 0:
                clear_output(wait=True)
                avg = self.plot_reward_history(title="Episode {} finished after {} timesteps".format(i_episode, t + 1))
            if i_episode % save_every == 0:
                torch.save(self.policy_net.state_dict(), "models/model.pt")

    def load_policy(self, path):
        """
        Load a policy from a file.
        """
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()

    @staticmethod
    def _update_average(old_average, num_obs, new_obs):
        new_average = old_average * num_obs / (num_obs + 1) + new_obs / (num_obs + 1)
        return new_average, num_obs + 1

    def plot_reward_history(self, title=""):
        """
        Plot the reward history to standard output.
        """
        import matplotlib.pyplot as plt
        averages = [self.reward_history[0]]
        for i in range(1, len(self.reward_history)):
            averages.append(self._update_average(averages[-1], i, self.reward_history[i])[0])
        plt.xlabel("Time")
        plt.ylabel("Running average of rewards")
        plt.title(title)
        plt.plot(averages, color="black")
        plt.show()
        return averages[-1]

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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            values = self.target_net(non_final_next_states)
            values = torch.mul(values, next_action_spaces_mask)
            next_state_values[non_final_mask] = values.max(1)[0].detach()
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Mean Squared Error loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the policy network
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        optimizer.step()
