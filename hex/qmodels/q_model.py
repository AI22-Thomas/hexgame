from abc import abstractmethod

import torch


class QModel(object):

    def __init__(self, input_size, output_size):
        self.policy_net = None
        self.target_net = None
        self.adv_net = None
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def make_network(self):
        pass

    def update_adv_net(self):
        self.adv_net.load_state_dict(self.policy_net.state_dict())

    def initialize_networks(self, device):
        self.policy_net = self.make_network().to(device)
        self.target_net = self.make_network().to(device)
        self.adv_net = self.make_network().to(device)
        self.update_adv_net()
        self.adv_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
        self.target_net.load_state_dict(torch.load(path))
        self.target_net.eval()
