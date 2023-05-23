from abc import abstractmethod

from hex.hex_engine import HexEngine

class HexEnvTransformer(object):

    @abstractmethod
    def dim_input(self, env):
        pass
    @abstractmethod
    def transform_board(self, env, engine: HexEngine, board):
        pass
