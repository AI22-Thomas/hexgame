import torch
import sys
from hex.hex_env import HexEnv
from hex.q_engine import QEngine
from hex.qmodels.conv_qmodel import ConvQModel
from hex.qmodels.simple_qmodel import SimpleQModel
from hex.transformers.conv_transformer import ConvTransfomer
from hex.transformers.simple_transformer import SimpleTransfomer

env = HexEnv(3,
             # transformer=ConvTransfomer()
             transformer=SimpleTransfomer()
             )
env.reset()

q_learner = QEngine(env,
                    # ConvQModel(env.dim_input(), env.dim_output())
                    SimpleQModel(env.dim_input(), env.dim_output())
                    )


# load newest model from models folder
q_learner.model.load_model("models/good2.pt")


def machine(board, action_set):
    board = env.transformer.transform_board(env, env.engine, board)
    board = torch.tensor(board, dtype=torch.float32, device=q_learner.device).unsqueeze(0)
    action_set = [env.engine.coordinate_to_scalar(x) for x in action_set]
    return env.engine.scalar_to_coordinates(q_learner._eps_greedy_action(board, 0, action_set))


env.engine.reset()
# env.engine.machine_vs_machine(machine, None)
env.engine.human_vs_machine(-1, machine)