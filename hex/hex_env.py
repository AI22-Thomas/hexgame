from hex_engine import HexEngine
from hex.transformers.hex_env_transformer import HexEnvTransformer


class HexEnv(object):
    def __init__(self, board_size, transformer: HexEnvTransformer):
        self.transformer = transformer
        self.board_size = board_size
        self.engine = HexEngine(board_size)

    def dim_input(self):
        return self.transformer.dim_input(self)

    def dim_output(self):
        return len(self.action_space())

    def reset(self):
        self.engine.reset()
        return self.transformer.transform_board(self, self.engine, self.engine.board), None

    def action_space(self):
        return [self.engine.coordinate_to_scalar(x) for x in
                self.engine.get_action_space(recode_black_as_white=self.engine.player == -1)]

    def step(self, action):
        action = self.engine.scalar_to_coordinates(action)

        if self.engine.player == -1:
            action = self.engine.recode_coordinates(action)

        self.engine.move(action)
        e_board = self.engine.board
        if self.engine.player == -1:
            e_board = self.engine.recode_black_as_white()

        reward = 0
        if self.engine.winner != 0:
            reward = 1 if self.engine.winner == 1 else -1

        return self.transformer.transform_board(self,
                                                self.engine,
                                                e_board), reward, self.engine.winner != 0, self.action_space()

    def close(self):
        pass
