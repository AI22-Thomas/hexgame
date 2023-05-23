import numpy as np

from hex.hex_env import HexEnv
from hex.transformers.hex_env_transformer import HexEnvTransformer


class ConvTransfomer(HexEnvTransformer):
    def dim_input(self, env: HexEnv):
        return env.board_size + 2

    class HexEnum:
        WHITE = 0
        WHITE_CONNECTED_LEFT = 1
        WHITE_CONNECTED_RIGHT = 2
        BLACK = 3
        BLACK_CONNECTED_TOP = 4
        BLACK_CONNECTED_BOTTOM = 5

    def _create_board(self, size):
        # Board is padded with 1 rows and 1 columns on each side
        board = []
        for row in range(size + 2):
            board.append([])
            for column in range(size + 2):
                # First two columns and last two are white
                # First two rows and last two are black
                black_top = 1 if row < 1 else 0
                black_bottom = 1 if row > size else 0
                white_left = 1 if column < 1 else 0
                white_right = 1 if column > size else 0
                black = 1 if black_top or black_bottom else 0
                white = 1 if white_left or white_right else 0
                board[row].append([white, white_left, white_right, black, black_top, black_bottom])
        return board

    def _convert_board(self, engine, engine_board, board_size):
        board = self._create_board(board_size)
        for i in range(len(engine_board)):
            for j in range(len(engine_board)):
                if engine_board[i][j] == -1:
                    board[i + 1][j + 1] = [0, 0, 0, 1, 0, 0]
                    # board[i + 2][j + 2] = HexEnum.BLACK
                    # Breadth first search to find connected black pieces

                elif engine_board[i][j] == 1:
                    board[i + 1][j + 1] = [1, 0, 0, 0, 0, 0]
                    # board[i + 2][j + 2] = HexEnum.WHITE

        def rec_walk(visited, position, hex_enum, color):
            if position in visited:
                return
            visited.add(position)
            adjacent = engine._get_adjacent(position)
            for adj in adjacent:
                if engine_board[adj[0]][adj[1]] == color:
                    board[adj[0] + 1][adj[1] + 1][hex_enum] = 1
                    # board[adj[0] + 2][adj[1] + 2] = hex_enum
                    rec_walk(visited, adj, hex_enum, color)

        # From all edges, find connected pieces

        for i in range(0, board_size):
            # Top edge
            if engine_board[0][i] == -1:
                board[1][i + 1][self.HexEnum.BLACK_CONNECTED_TOP] = 1
                # board[2][i + 2] = HexEnum.BLACK_CONNECTED_TOP
                visited = {0, i}
                rec_walk(visited, (0, i), self.HexEnum.BLACK_CONNECTED_TOP, -1)
            # Bottom edge
            if engine_board[board_size - 1][i] == -1:
                board[board_size][i + 1][self.HexEnum.BLACK_CONNECTED_BOTTOM] = 1
                # board[board_size + 1][i + 2] = self.HexEnum.BLACK_CONNECTED_BOTTOM
                visited = {board_size - 1, i}
                rec_walk(visited, (board_size - 1, i), self.HexEnum.BLACK_CONNECTED_BOTTOM, -1)
            # Left edge
            if engine_board[i][0] == 1:
                board[i + 1][1][self.HexEnum.WHITE_CONNECTED_LEFT] = 1
                # board[i + 2][2] = self.HexEnum.WHITE_CONNECTED_LEFT
                visited = {i, 0}
                rec_walk(visited, (i, 0), self.HexEnum.WHITE_CONNECTED_LEFT, 1)
            # Right edge
            if engine_board[i][board_size - 1] == 1:
                board[i + 1][board_size][self.HexEnum.WHITE_CONNECTED_RIGHT] = 1
                # board[i + 2][board_size + 1] = self.HexEnum.WHITE_CONNECTED_RIGHT
                visited = {i, board_size - 1}
                rec_walk(visited, (i, board_size - 1), self.HexEnum.WHITE_CONNECTED_RIGHT, 1)

        return board

    def transform_board(self, env, engine, board):
        board = self._convert_board(engine, board, env.board_size)
        board = np.array(board)
        board = board.transpose((2, 0, 1))
        return board
