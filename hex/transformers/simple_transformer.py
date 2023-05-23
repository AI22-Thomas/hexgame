import numpy as np

from hex.hex_env import HexEnv
from hex.transformers.hex_env_transformer import HexEnvTransformer


class SimpleTransfomer(HexEnvTransformer):
    def dim_input(self, env: HexEnv):
        return env.board_size ** 2

    def transform_board(self, env, engine, board):
        return np.array(board).flatten()
