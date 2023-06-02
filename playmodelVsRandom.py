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

# load newest model from models folder
q_learner.model.load_model("models/model.pt")

def machine(board, action_set):
    board = env.transformer.transform_board(env, env.engine, board)
    board = torch.tensor(board, dtype=torch.float32, device=q_learner.device).unsqueeze(0)
    action_set = [env.engine.coordinate_to_scalar(x) for x in action_set]
    return env.engine.scalar_to_coordinates(q_learner._eps_greedy_action(board, 0, action_set))


def b_machine(board, action_set):
    board = env.transformer.transform_board(env, env.engine, env.engine.recode_black_as_white())
    board = torch.tensor(board, dtype=torch.float32, device=q_learner.device).unsqueeze(0)
    action_set = [env.engine.coordinate_to_scalar(env.engine.recode_coordinates(x)) for x in action_set]
    return env.engine.recode_coordinates(
        env.engine.scalar_to_coordinates(q_learner._eps_greedy_action(board, 0, action_set)))


def b_straight(board, action_set):
    for action in action_set:
        if action[1] == 6:
            return action
    return action_set[0]
import os

wp_black_wins = 0
wp_white_wins = 0

black_wins = 0
white_wins = 0
for i in range(100000):
    env.engine.reset()    
    env.engine.machine_vs_machine(machine, None, verbose=False)

    if env.engine.winner == -1:
        wp_black_wins += 1
    else:
        wp_white_wins += 1
    totWinsW = (wp_black_wins) + (wp_white_wins)
    ranWinPercW = (wp_black_wins)/totWinsW
    aiWinPercW = (wp_white_wins)/totWinsW

    env.engine.machine_vs_machine(None, b_machine, verbose=False)
    if env.engine.winner == -1:
        black_wins += 1
    else:
        white_wins += 1

    totWins = (black_wins) + (white_wins)
    aiWinPerc = (black_wins)/totWins

    ranWinPerc = (white_wins)/totWins

    if i% 250 == 0: 
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"As BLACK: AI wins: {black_wins} ; Random wins: {white_wins}, {aiWinPerc}, {ranWinPerc}\nAs WHITE: AI wins: {wp_white_wins} ; Random wins: {wp_black_wins}, {aiWinPercW}, {ranWinPercW}",end='/r', flush=True)

    #continue on enter
    if(i % 10000 == 0):
      input()
    
# env.engine.machine_vs_machine(machine, b_machine)
# env.engine.human_vs_machine(-1, machine)
