import numpy as np
from typing import Tuple, List
import logging


class GameEnv():

    def __init__(self) -> None:
        # dimension of the board
        self.N_rows: int = 3
        self.N_cols: int = 3
        # how many filled cell in a row to win
        self.N_to_win: int = 3
        self.starting_player: int = 1
        # board
        self.board: np.ndarray[Tuple[int, int], np.dtype[np.int8]]
        self.current_player: int
        self.is_game_over: bool
        self.winner: int|None
        self.reset_game()

    def reset_game(self) -> None:
        # board
        self.board = np.zeros((self.N_rows, self.N_cols), dtype=np.int8)
        # player
        self.current_player: int = self.starting_player
        # indicate if game is over
        self.is_game_over: bool = False
        self.winner: int|None = None
    
    def change_state(self, action: Tuple[int, int]) -> None:
        """change board state to new state after action of player

        Args:
            action (Tuple[int, int]): action
        """
        if self.current_player == 1:
            self.board[action] = 1
        elif self.current_player == -1:
            self.board[action] = -1
        else:
            logging.error(f"player {self.current_player} is invalid. Player should be 1 or -1")

    def check_winner(self, matches_player: np.ndarray[Tuple[int], np.dtype[np.int8]]) -> None:
        # Create a kernel of N ones
        kernel = np.ones(self.N_to_win, dtype=np.int8)
        # Perform convolution
        # If N_to_win consecutive True values exist, their sum in convolution will be N
        convolved_result = np.convolve(matches_player, kernel, mode='valid')
        # Check if any part of the convolution result equals N
        if np.any(convolved_result == self.N_to_win):
            self.winner = self.current_player
            self.is_game_over = True

    def get_significant_diagonals(self) -> List[np.ndarray[Tuple[int], np.dtype[np.int8]]]:
        """
        Extracts diagonals from a 2D NumPy array that have at least a specified minimum length=self.N_to_win

        Returns:
            A list of 1D NumPy arrays, each representing a significant diagonal.
        """
        diagonals = []

        # --- Regular Diagonals (top-left to bottom-right) ---
        # Iterate through possible offsets.
        # The range for offsets goes from -(rows - 1) to (cols - 1).
        # We only care about those whose length is >= min_length.
        for offset in range(-(self.N_rows - 1), self.N_cols):
            diag = np.diagonal(self.board, offset=offset)
            if len(diag) >= self.N_to_win:
                diagonals.append(diag)

        # --- Anti-Diagonals (top-right to bottom-left) ---
        # Flip the board horizontally and apply the same logic.
        flipped_board = np.fliplr(self.board)
        for offset in range(-(self.N_rows - 1), self.N_cols):
            diag = np.diagonal(flipped_board, offset=offset)
            if len(diag) >= self.N_to_win:
                diagonals.append(diag)

        return diagonals

    def check_game_state(self):
        # check if all cells are filled
        if np.sum(self.board == 0) == 0:
            self.is_game_over = True
        # check if player 1 wins
        ## columns
        for r in range(self.N_rows):
            matches_player = (self.board[r] == self.current_player) # Boolean array: True where player is present
            self.check_winner(matches_player)
        ## rows
        for r in range(self.N_cols):
            matches_player = (self.board[:,r] == self.current_player) # Boolean array: True where player is present
            self.check_winner(matches_player)
        ## diag
        diagonals = self.get_significant_diagonals()
        for diag in diagonals:
            matches_player = (diag == self.current_player)
            self.check_winner(matches_player)

    def step(self, action: Tuple[int, int]):
        # 1. check if action is valid
        if (~(0 <= action[0] < self.N_rows)) or (~(0 <= action[1] < self.N_rows)):
            logging.error(f"action {action} is invalid")
        # 2. change state of board using action
        self.change_state(action)
        # 3. check if winner or game over
        self.check_game_state()
        # 4. if game not over and no winner, then switch player
        if self.is_game_over:
            if self.winner == None:
                print("game is over and there is no winner")
                logging.info("game is over and there is no winner")
            else:
                print(f"game is over and winner is player {self.winner}")
                logging.info(f"game is over and winner is player {self.winner}")
        else:
            self.current_player -= self.current_player * 2

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """
        Returns a list of all valid (empty) cell coordinates on the board.
        This uses a vectorized NumPy operation for efficiency.
        """
        # Find all coordinates where the board value is 0 (empty)
        empty_cell_coordinates = np.argwhere(self.board == 0)

        # Convert the NumPy array of coordinates to a list of Python tuples
        # Each 'coord' in empty_cell_coordinates will be a small 1D array like [row, col]
        valid_moves = [(np.int8(coord[0]), np.int8(coord[1])) for coord in empty_cell_coordinates]
        return valid_moves
    
    def random_model(self, ind_valid: List[Tuple[int, int]]) -> Tuple[int, int]:
        N_valid_actions = len(ind_valid)
        choice = np.random.randint(low=0, high=N_valid_actions, size=1)
        return ind_valid[choice[0]]
    
    def run_game(self):
        while not self.is_game_over:
            ind_valid = self.get_valid_actions()
            action = self.random_model(ind_valid)
            self.step(action)








if __name__=="__main__":
    print("hello")
    env = GameEnv()
    env.run_game()
    print(env.board)