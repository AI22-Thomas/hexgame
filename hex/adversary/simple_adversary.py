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
        self.runsAll =0
        self.netChanges = 0
        self.averages = []
    def init(self, q_learner):
        self.net = q_learner.model.make_network().to(q_learner.device)
        self.net.load_state_dict(q_learner.model.policy_net.state_dict())
        self.net.eval()

    def update(self, q_learner, epoch, showPlot=False):
        snaps = os.listdir("models/snaps")

        if epoch == 0:
            self.netChanges +=1
            self.net.load_state_dict(q_learner.model.policy_net.state_dict())
            self.net.eval()
            return
            #check if snaps in folder
            if len(snaps) == 0:
                print("No snaps found in folder")
                #load from qlearner
                self.net.load_state_dict(q_learner.model.policy_net.state_dict())
                self.net.eval()
                return
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

            #average reward
            avg_rewW = sum(rewardsW) / len(rewardsW)
            avg_rewB = sum(rewardsB) / len(rewardsB)

            #avg reward total 
            avg_rew = (avg_rewW + avg_rewB) / 2
            print("Avg. Reward t, w, b: ", avg_rew, avg_rewW, avg_rewB)
            self.runs +=1;
            if avg_rew > self.update_threshold:
                self.runsAll +=1;
                print("Updated adversary at epoch", epoch)
                self.netChanges +=1
                if(self.runsAll >8):
                    self.runsAll = 0
                    ##play against all models in models/snaps
                    #snaps = os.listdir("models/snaps")
                    #snaps.sort(key=lambda x: float(x.split("_")[1].split(".")[0]))
                    #allAverages = []
                    #for snap in snaps:
                    #    self.net.load_state_dict(torch.load("models/snaps/" + snap))
                    #    self.net.eval()
                    #    rewardsW = q_learner.play(q_learner.env, play_as_black=False, randomColorOff=True, playWithRandomStart=True)
                    #    rewardsB = q_learner.play(q_learner.env, play_as_black=True, randomColorOff=True, playWithRandomStart=True)
                    #    print("Updated adversary at epoch", epoch)
                    #    #average reward
                    #    avg_rewW = sum(rewardsW) / len(rewardsW)
                    #    avg_rewB = sum(rewardsB) / len(rewardsB)
                    #    avg_rew = (avg_rewW + avg_rewB) / 2
                    #    print("Model: ", snap, "Avg. Reward t, w, b: ", avg_rew, avg_rewW, avg_rewB)
                    #    allAverages.append (avg_rew)
                    
                    #self.averages.append(allAverages)
                    ##save all averages to txt file to open later as plot
                    #with open("models/averages.txt", "a") as f:
                    #    f.write(str(allAverages) + "\n")
                        
                    ##plot all 'allAverages' in self.averages 
                    #for averagesA in self.averages:
                    #    plt.plot(averagesA)
                    #    plt.legend()
                    #    #x titel model
                    #    #y titel avg reward
                    #    plt.xlabel("Model")
                    #    plt.ylabel("Avg. Reward")
                    #    plt.show()

                    
                #update adversary model with current QLearning model or randomly a model from models/snaps
                if len(snaps) == 0:
                    print("Start... changed to current model")
                    self.net.load_state_dict(q_learner.model.policy_net.state_dict())     
                else:  
                    #update adversary model with current QLearning model or randomly a model from models/snaps
                    if self.netChanges == 3:
                        print("Changed to current model")
                        self.netChanges = 0
                        self.net.load_state_dict(q_learner.model.policy_net.state_dict())
                    else:
                        #get all files in snap folder
                        #sort by timestamp (split filename and sort by timestamp)
                        snaps.sort(key=lambda x: float(x.split("_")[1].split(".")[0]))
                        #load random model
                        snap = snaps[random.randint(0, len(snaps) - 1)]
                        self.net.load_state_dict(torch.load("models/snaps/" + snap))
                        print("changed to model: ", snap)
                        
                #save model
                if(self.runs > 1):
                    torch.save(q_learner.model.policy_net.state_dict(), "models/snaps/model_{}.pt".format(time.time()))
                    print("Saved Model at: ", self.runs)
                    print("-------------------- starting Test Runs --------- ", self.runs)

                    #play against all models in models/snaps
                    snaps = os.listdir("models/snaps")
                    snaps.sort(key=lambda x: float(x.split("_")[1].split(".")[0]))
                    allAverages = []
                    for snap in snaps:
                        self.net.load_state_dict(torch.load("models/snaps/" + snap))
                        self.net.eval()
                        rewardsW = q_learner.play(q_learner.env, play_as_black=False, randomColorOff=True, playWithRandomStart=True)
                        rewardsB = q_learner.play(q_learner.env, play_as_black=True, randomColorOff=True, playWithRandomStart=True)
                        avg_rewW = sum(rewardsW) / len(rewardsW)
                        avg_rewB = sum(rewardsB) / len(rewardsB)
                        avg_rew = (avg_rewW + avg_rewB) / 2
                        print("Tested against Model: ", snap, "Avg. Reward t, w, b: ", avg_rew, avg_rewW, avg_rewB)
                        allAverages.append (avg_rew)

                    print("-------------------- Ended Test Runs --------- ", self.runs)
                    
                    self.averages.append(allAverages)
                    #save all averages to txt file to open later as plot
                    with open("models/averages.txt", "a") as f:
                        f.write(str(allAverages) + "\n")

                self.runs = 0;
                self.net.eval()
                    
                
    def get_action(self, state, q_learner):
        return q_learner._eps_greedy_action(
            state,
            eps=0,
            net=self.net)
