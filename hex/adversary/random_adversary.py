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
        #Cheeck iuf model is better or worse than before (trained model vs current adversary)
        rewardsW = q_learner.play(q_learner.env, play_as_black=False, randomColorOff=True, playWithRandomStart=random_start, printBoard=False)
        rewardsB = q_learner.play(q_learner.env, play_as_black=True, randomColorOff=True, playWithRandomStart=random_start, printBoard=False)

        #average reward
        avg_rewW = sum(rewardsW) / len(rewardsW)
        avg_rewB = sum(rewardsB) / len(rewardsB)

        #avg reward total 
        avg_rew = (avg_rewW + avg_rewB) / 2
        print("Avg. Reward t, w, b: ", avg_rew, avg_rewW, avg_rewB)