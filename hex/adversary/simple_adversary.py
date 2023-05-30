import os
import time

import matplotlib.pyplot as plt
import torch
from hex.adversary.base_adversary import BaseAdversary
from hex.qmodels.q_model import QModel

import random


class SimpleAdversary(BaseAdversary):

    def __init__(self, update_threshold,
                 check_interval
                 ):
        super().__init__()
        self.net = None
        self.update_threshold = update_threshold
        self.check_interval = check_interval
        self.runs = 0

    def init(self, q_learner):
        self.net = q_learner.model.make_network().to(q_learner.device)
        self.net.load_state_dict(q_learner.model.policy_net.state_dict())
        self.net.eval()

    def update(self, q_learner, epoch, showPlot=False):

        if epoch == 0:
            snaps.sort(key=lambda x: float(x.split("_")[1].split(".")[0]))
            #load random model
            snap = snaps[random.randint(0, len(snaps) - 1)]
            self.net.load_state_dict(torch.load("models/snaps/" + snap))
            print("changed to model: ", snap)            
            self.net.eval()
            print("Updated adversary at epoch 0")
            return

        if epoch % self.check_interval == 0:
            #Cheeck iuf model is better or worse than before (trained model vs current adversary)
            rewardsW = q_learner.play(q_learner.env, 100, play_as_black=False, randomColorOff=True, playWithRandomStart=True)
            rewardsB = q_learner.play(q_learner.env, 100, play_as_black=True, randomColorOff=True, playWithRandomStart=True)

            if showPlot:
                #plot rewardsW & rewardsW
                plt.plot(rewardsW, label="White")
                plt.plot(rewardsB, label="Black")
                plt.legend()
                plt.show()
            
            #average reward
            avg_rewW = sum(rewardsW) / len(rewardsW)
            avg_rewB = sum(rewardsB) / len(rewardsB)

            #avg reward total 
            avg_rew = (avg_rewW + avg_rewB) / 2
            print("Avg. Reward t, w, b: ", avg_rew, avg_rewW, avg_rewB)
            self.runs +=1;
            if avg_rew > self.update_threshold:
                print("Updated adversary at epoch", epoch)
                #change to model 
                ## get all files in snap folder
                #snaps = os.listdir("models/snaps")
                ## sort by timestamp (split filename and sort by timestamp)
                #snaps.sort(key=lambda x: float(x.split("_")[1].split(".")[0]))
                #for i in range(len(snaps) - 5):
                #    os.remove("models/snaps/" + snaps[i])
                # save model with timestamp
                
                snaps = os.listdir("models/snaps")
                #check if items in snaps
                if len(snaps) == 0:
                    print("Start... changed to current model")
                    self.net.load_state_dict(q_learner.model.policy_net.state_dict())     
                else:  
                    #update adversary model with current QLearning model or randomly a model from models/snaps
                    if random.random() < 0.25:
                        print("Changed to current model")
                        self.net.load_state_dict(q_learner.model.policy_net.state_dict())
                    else:
                        #get all files in snap folder
                        #sort by timestamp (split filename and sort by timestamp)
                        snaps.sort(key=lambda x: float(x.split("_")[1].split(".")[0]))
                        #load random model
                        snap = snaps[random.randint(0, len(snaps) - 1)]
                        self.net.load_state_dict(torch.load("models/snaps/" + snap))
                        print("changed to model: ", snap)
                if(self.runs > 1):
                    torch.save(q_learner.model.policy_net.state_dict(), "models/snaps/model_{}.pt".format(time.time()))
                    print("Saved Model at: ", self.runs)
                self.runs = 0;
                self.net.eval()
                    
                
    def get_action(self, state, q_learner):
        return q_learner._eps_greedy_action(
            state,
            eps=0,
            net=self.net)
