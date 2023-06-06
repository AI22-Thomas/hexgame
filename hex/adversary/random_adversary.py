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
        q_learner.model.policy_net.eval()
        if epoch % 500 == 0:
            #Cheeck iuf model is better or worse than before (trained model vs current adversary)
            rewardsW = q_learner.play(q_learner.env, 100, play_as_black=False, randomColorOff=True, playWithRandomStart=random_start, printBoard=False)
            rewardsB = q_learner.play(q_learner.env, 100, play_as_black=True, randomColorOff=True, playWithRandomStart=random_start, printBoard=False)

            avg_rewW = sum(rewardsW) / len(rewardsW)
            avg_rewB = sum(rewardsB) / len(rewardsB)

            avg_rew = (avg_rewW + avg_rewB) / 2
            
            print("Random adversary avg reward: ", avg_rew, ", White: ", avg_rewW, ", Black: ", avg_rewB)
        q_learner.model.policy_net.train()
