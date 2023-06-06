import torch
import sys

from hex.adversary.random_adversary import RandomAdversary
from hex.adversary.simple_adversary import SimpleAdversary

from hex.hex_env import HexEnv
from hex.q_engine import QEngine
from hex.qmodels.conv_qmodel import ConvQModel
from hex.qmodels.simple_qmodel import SimpleQModel
from hex.transformers.conv_transformer import ConvTransfomer
from hex.transformers.simple_transformer import SimpleTransfomer

BOARD_SIZE = 7

env = HexEnv(BOARD_SIZE,
             # transformer=ConvTransfomer()
             transformer=SimpleTransfomer()
             )
env.reset()

q_learner = QEngine(env,
                    # ConvQModel(env.dim_input(), env.dim_output())
                    SimpleQModel(env.dim_input(), env.dim_output()),
                  #adversary=RandomAdversary(),
                    adversary=SimpleAdversary(update_threshold=0.915,check_interval=512),
                    )

q_learner.model.load_model("models/model.pt")


def machine(board, action_set):
    board = env.transformer.transform_board(env, env.engine, board)
    board = torch.tensor(board, dtype=torch.float32, device=q_learner.device).unsqueeze(0)
    action_set = [env.engine.coordinate_to_scalar(x) for x in action_set]
    return env.engine.scalar_to_coordinates(q_learner._eps_greedy_action(board, 0, action_set).item())


def b_machine(board, action_set):
    board = env.transformer.transform_board(env, env.engine, env.engine.recode_black_as_white())
    board = torch.tensor(board, dtype=torch.float32, device=q_learner.device).unsqueeze(0)
    action_set = [env.engine.coordinate_to_scalar(env.engine.recode_coordinates(x)) for x in action_set]
    return env.engine.recode_coordinates(
        env.engine.scalar_to_coordinates(q_learner._eps_greedy_action(board, 0, action_set).item()))


def b_straight(board, action_set):
    for action in action_set:
        if action[1] == 6:
            return action
    return action_set[0]

import os

# Get user input
user_input = input("Enter either 'ran' or 'snaps': ")

# Determine if snapsRan based on user input
snapsRan = user_input.lower() == 'ran'

if(snapsRan):
    folderPath = 'snapsRandom'
else:
    folderPath = 'snaps'

#play against all models in models/snaps
snaps = os.listdir("models/"+folderPath)
#add models/snapsRandom to snaps

snaps.sort(key=lambda x: float(x.split("_")[1].split(".")[0]))
allAverages = []
for snap in snaps:
    print(snap)
    q_learner.adversary.net.load_state_dict(torch.load("models/"+folderPath+"/" + snap))
    q_learner.adversary.net.eval()
    
    rewardsW = q_learner.play(q_learner.env, 1,play_as_black=False, randomColorOff=True, playWithRandomStart=True)
    rewardsB = q_learner.play(q_learner.env, 1,play_as_black=True, randomColorOff=True, playWithRandomStart=True)
    avg_rewW = sum(rewardsW) / len(rewardsW)
    avg_rewB = sum(rewardsB) / len(rewardsB)
    avg_rew = (avg_rewW + avg_rewB) / 2
    #print("Tested against Model: ", snap, "Avg. Reward t, w, b: ", avg_rew, avg_rewW, avg_rewB)
    allAverages.append (avg_rew)

arrayOfAverages = allAverages
print(arrayOfAverages)
av = sum(arrayOfAverages)/len(arrayOfAverages)
print("Average: ", av)
#averages = [sum(array) / len(array) if len(array) > 0 else 0 for array in arrayOfAverages]
#print("Av1: ", arrayOfAverages)
##plot averages as a line graph
#plt.plot(arrayOfAverages)
#plt.ylabel('averages')
#plt.xlabel('Episode')
#plt.show()



input()
#black_wins = 0
#white_wins = 0
#for i in range(800):
#    env.engine.reset()    
#    env.engine.machine_vs_machine(machine, b_machine)
#    #env.engine.machine_vs_machine(None, b_machine)
#    #env.engine.human_vs_machine(human_player=1, machine=b_machine)
#    #env.engine.machine_vs_machine(None, b_machine)
#    if env.engine.winner == -1:
#        black_wins += 1
#    else:
#        white_wins += 1

#    print("Black wins: ", black_wins)
#    print("White wins: ", white_wins)
#    #continue on enter
#    input()    
# env.engine.machine_vs_machine(machine, b_machine)
# env.engine.human_vs_machine(-1, machine)
