import random

import torch

from hex.adversary.base_adversary import BaseAdversary


class RandomAdversary(BaseAdversary):
    def get_action(self, state, q_learner):
        return torch.tensor([random.sample(q_learner.env.action_space(), 1)], device=q_learner.device,
                            dtype=torch.long)

    def init(self, q_learner):
        pass

    def update(self, q_learner, epoch,showPlot=False, random_start=False):
        pass
