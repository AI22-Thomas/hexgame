from collections import OrderedDict

import torch.nn as nn

from hex.qmodels.q_model import QModel


class SimpleQModel(QModel):

    def _make_network(self):
        layers = []
        layer_num = 0
        layers.append((str(layer_num), nn.Linear(self.input_size, 128)))
        layer_num += 1
        layers.append((str(layer_num), nn.ReLU()))
        layer_num += 1
        for i in range(2):
            layers.append((str(layer_num), nn.Linear(128, 128)))
            layer_num += 1
            layers.append((str(layer_num), nn.ReLU()))
            layer_num += 1
        layers.append((str(layer_num), nn.Linear(128, self.output_size)))
        layer_num += 1
        layers.append((str(layer_num), nn.Sigmoid()))
        net = nn.Sequential(OrderedDict(layers))
        return net
