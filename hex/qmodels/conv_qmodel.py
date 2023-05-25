import torch

import torch.nn as nn

from hex.qmodels.q_model import QModel


class ConvQModel(QModel):

    def make_network(self):
        net = nn.Sequential()
        conv = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        # 3x3 weights with 6 channels
        weights = torch.zeros(32, 6, 3, 3, dtype=torch.float32)
        # set weights for all channels to this:
        # 0 1 1
        # 1 1 1
        # 1 1 0
        for i in range(6):
            weights[:, i, :, :] = torch.tensor([[0, 1, 1],
                                                [1, 1, 1],
                                                [1, 1, 0]], dtype=torch.float32)
        with torch.no_grad():
            conv.weight = nn.Parameter(weights)

        net.add_module("conv", conv)
        net.add_module("relu1", nn.ReLU())
        net.add_module("flatten", nn.Flatten())
        net.add_module("fc1", nn.Linear(32 * self.input_size * self.input_size, 128))
        net.add_module("relu2", nn.ReLU())
        net.add_module("fc2", nn.Linear(128,128))
        net.add_module("relu3", nn.ReLU())
        net.add_module("fc3", nn.Linear(128, self.output_size))
        net.add_module("sigmoid", nn.Sigmoid())
        return net
