import sys
from hex.hex_env import HexEnv
from hex.q_engine import QEngine
from hex.qmodels.conv_qmodel import ConvQModel
from hex.qmodels.simple_qmodel import SimpleQModel
from hex.transformers.conv_transformer import ConvTransfomer
from hex.transformers.simple_transformer import SimpleTransfomer

BOARD_SIZE = 5

env = HexEnv(BOARD_SIZE,
             # transformer=ConvTransfomer()
             transformer=SimpleTransfomer()
             )
env.reset()

q_learner = QEngine(env,
                    # ConvQModel(env.dim_input(), env.dim_output())
                    SimpleQModel(env.dim_input(), env.dim_output())
                    )
q_learner.learn(batch_size=64,
                num_episodes=2000000,
                eps_start=0.1,
                eps_end=0.1,
                eps_decay=1,
                gamma=1,
                target_net_update_rate=0.001,
                soft_update=True,
                # soft_update=False,
                # target_net_update_rate=100,
                learning_rate=0.0001,
                eval_every=100,
                save_every=100,
                random_start=True,
                self_play=False,
                start_from_model="models/model.pt",
                save_path="models/model.pt",
                # start_from_model="models/model_conv.pt",
                # save_path="models/model_conv.pt",
                evaluate_runs=100
                )