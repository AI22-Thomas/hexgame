import torch
import sys

from hex.adversary.random_adversary import RandomAdversary
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
                    adversary=RandomAdversary(),
                    )

import os
import glob
import torch

# Get list of all .pt files in the directory
model_files = glob.glob("models/snaps/*.pt")

# Sort files by modification time in descending order
model_files.sort(key=os.path.getmtime, reverse=True)

# Take the first (newest) file
#latest_model_file = model_files[0]
latest_model_file = "models/model.pt"

# Load the model
q_learner.model.load_model(latest_model_file)
q_learner.model.policy_net.eval()

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


black_wins = 0
white_wins = 0
for i in range(800):
    env.engine.reset()    
    env.engine.machine_vs_machine(machine, b_machine, printBoard=True, verbose=True)
    #env.engine.machine_vs_machine(None, b_machine)
    #env.engine.human_vs_machine(human_player=1, machine=b_machine)
    #env.engine.machine_vs_machine(None, b_machine)
    if env.engine.winner == -1:
        black_wins += 1
    else:
        white_wins += 1

    print("Black wins: ", black_wins)
    print("White wins: ", white_wins)
    #continue on enter
    input()    
# env.engine.machine_vs_machine(machine, b_machine)
# env.engine.human_vs_machine(-1, machine)
