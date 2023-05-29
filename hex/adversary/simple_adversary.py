import os
import time

import torch

from hex.adversary.base_adversary import BaseAdversary
from hex.qmodels.q_model import QModel


class SimpleAdversary(BaseAdversary):

    def __init__(self, update_threshold,
                 check_interval,
                 check_runs):
        super().__init__()
        self.net = None
        self.update_threshold = update_threshold
        self.check_interval = check_interval
        self.check_runs = check_runs

    def init(self, q_learner):
        self.net = q_learner.model.make_network().to(q_learner.device)
        self.net.load_state_dict(q_learner.model.policy_net.state_dict())
        self.net.eval()

    def update(self, q_learner, epoch):

        if epoch == 0:
            self.net.load_state_dict(q_learner.model.policy_net.state_dict())
            self.net.eval()
            print("Updated adversary at epoch 0")
            return

        if epoch % self.check_interval == 0:
            rewards = q_learner.play(q_learner.env, self.check_runs)
            avg_rew = sum(rewards) / len(rewards)
            if avg_rew > self.update_threshold:
                self.net.load_state_dict(q_learner.model.policy_net.state_dict())
                self.net.eval()
                print("Updated adversary at epoch", epoch)

                # get all files in snap folder
                snaps = os.listdir("models/snaps")
                # sort by timestamp (split filename and sort by timestamp)
                snaps.sort(key=lambda x: float(x.split("_")[1].split(".")[0]))
                for i in range(len(snaps) - 5):
                    os.remove("models/snaps/" + snaps[i])
                # save model with timestamp
                torch.save(q_learner.model.policy_net.state_dict(), "models/snaps/model_{}.pt".format(time.time()))

    def get_action(self, state, q_learner):
        return q_learner._eps_greedy_action(
            state,
            eps=0,
            net=self.net)
