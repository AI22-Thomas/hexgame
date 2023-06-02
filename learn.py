import sys

from hex.adversary.random_adversary import RandomAdversary
from hex.adversary.simple_adversary import SimpleAdversary

sys.path.append("/home/ai22m008/.local/lib/python3.8/site-packages/")
from hex.hex_env import HexEnv
from hex.q_engine import QEngine
from hex.qmodels.conv_qmodel import ConvQModel
from hex.qmodels.simple_qmodel import SimpleQModel
from hex.transformers.conv_transformer import ConvTransfomer
from hex.transformers.simple_transformer import SimpleTransfomer

BOARD_SIZE = 7

env = HexEnv(BOARD_SIZE,
             # transformer=ConvTransfomer(),
             transformer=SimpleTransfomer()
             )
env.reset()

q_learner = QEngine(env,
                    # ConvQModel(env.dim_input(), env.dim_output()),
                    SimpleQModel(env.dim_input(), env.dim_output()),
                    chart=False,  # Whether to plot a chart of the rewards
                    # random play
                    adversary=RandomAdversary(),
                    # self play
                    #adversary=SimpleAdversary(update_threshold=0.915,check_interval=512),
                    )
q_learner.learn(batch_size=64,
                num_episodes=2500000,
                eps_start=0.1,
                eps_end=0.1,
                eps_decay=1,
                gamma=0.95,
                # target_net_update_rate=0.001,
                # soft_update=True,
                soft_update=False,
                target_net_update_rate=128,
                learning_rate=0.00042,
                eval_every=1048,
                save_every=128,
                random_start=False,
                start_from_model="models/model.pt",
                save_path="models/model.pt",
                # start_from_model="models/model_conv.pt",
                # save_path="models/model_conv.pt",
                evaluate_runs=128,
                clip_grads=8,
                playAsColor=0.5
                )
