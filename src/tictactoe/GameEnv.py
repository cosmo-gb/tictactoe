import numpy as np
from typing import Tuple, List, Dict
import logging
import pickle


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
        self.reward: int
        self.reset()

    def reset(self) -> np.ndarray[Tuple[int, int], np.dtype[np.int8]]:
        # board
        self.board = np.zeros((self.N_rows, self.N_cols), dtype=np.int8)
        # player
        self.current_player: int = self.starting_player
        # indicate if game is over
        self.is_game_over: bool = False
        self.winner: int|None = None
        self.reward = 0
        return self.board
    
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
        # check if player wins
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

    def check_valid_action(self, action: Tuple[int]) -> bool:
        if self.board[action] != 0:
            return False
        return True
            
    def step(self, action: Tuple[int, int]):
        # 1. check if action is valid
        if (~(0 <= action[0] < self.N_rows)) or (~(0 <= action[1] < self.N_rows)):
            logging.error(f"Invalid: action {action} can not be done in this board.")
        if not self.check_valid_action(action):
            logging.error(f"Invalid: action {action} has been played already.")
        # 2. change state of board using action
        self.change_state(action)
        # 3. check if winner or game over
        self.check_game_state()
        # 4. if game not over and no winner, then switch player
        if self.is_game_over:
            if self.winner == None:
                logging.info("game is over and there is no winner")
                self.reward = 0
            else:
                logging.info(f"game is over and winner is player {self.winner}")
                if self.winner == 1:
                    self.reward = 1
                else:
                    self.reward = -1
        else:
            self.reward = 0
            self.current_player -= self.current_player * 2

        return self.board, self.reward, self.is_game_over

    def get_valid_actions(self, state: np.ndarray[Tuple[int, int], np.dtype[np.int8]]) -> List[Tuple[int, int]]:
        """
        Returns a list of all valid (empty) cell coordinates on the board.
        This uses a vectorized NumPy operation for efficiency.
        """
        # Find all coordinates where the board value is 0 (empty)
        empty_cell_coordinates = np.argwhere(state == 0)

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
            ind_valid = self.get_valid_actions(state=self.board)
            action = self.random_model(ind_valid)
            self.step(action)


    def render(self) -> None:
        """
        Prints a human-readable representation of the current game board to the console.
        'X' represents Player 1, 'O' represents Player -1, '_' represents an empty cell.
        """
        # Step 1: Define the mapping from internal board values to display symbols.
        # Use a Python dictionary for this mapping.
        symbols = {
            0: '_',   # Represents an empty cell
            1: 'X',   # Represents Player 1's mark
            -1: 'O'   # Represents Player -1's (opponent's) mark
        }

        # Step 2: Iterate through each row of the board.
        # Use a 'for' loop with 'range(self.N_rows)'.
        for r_idx in range(self.N_rows):
            # Step 2.1: Initialize an empty list to hold the symbols for the current row.
            row_display_elements = []

            # Step 2.2: Iterate through each column in the current row.
            # Use another nested 'for' loop with 'range(self.N_cols)'.
            for c_idx in range(self.N_cols):
                # Step 2.2.1: Get the numerical value from the board cell.
                # Access the NumPy array element using 'self.board[r_idx, c_idx]'.
                cell_value = self.board[r_idx, c_idx]

                # Step 2.2.2: Look up the corresponding display symbol.
                # Use the 'symbols' dictionary defined earlier.
                display_symbol = symbols[cell_value]

                # Step 2.2.3: Add the symbol to the list for the current row.
                # Use the 'append()' method of the list.
                row_display_elements.append(display_symbol)
            
            # Step 2.3: Join the symbols with a vertical separator and print the row.
            # Use the 'str.join()' method. The separator will be " | ".
            print(" | ".join(row_display_elements))

            # Step 2.4: Print a horizontal separator line between rows.
            # This should only happen if it's not the last row.
            # Use an 'if' statement: 'if r_idx < self.N_rows - 1:'
            if r_idx < self.N_rows - 1:
                # The separator string should have '---' for each column,
                # separated by '+'. For a 3x3 board, it's "---+---+---".
                # If N_cols were dynamic, you'd generate this string based on N_cols.
                print("---+---+---")

        # Step 3: Print an extra newline character for better spacing after the entire board.
        print("\n")




class Agent():

    def __init__(self, game_env_instance) -> None:

        #self.initial_epsilon = 1
        #self.epsilon_decay_rate = 0.99
        #self.final_epsilon = 0.1
        self.epsilon = 1 #self.initial_epsilon

        # determines how much future rewards are valued compared to immediate rewards
        # discount_factor = 0 means only immediate rewards matters
        # discount_factor = 1 means future and immediate rewards matter equally
        self.discount_factor: float = 0.9
        # as in deep learning, it controls how much old values are updated
        # learning_rate = 0 means learning nothing
        # learning_rate = 1 means completely forgot what we learnt before this reward
        self.learning_rate: float = 0.1

        self.q_table: Dict[Tuple[int], Dict[Tuple[int, int], float]] = {}
        self.nothing_q_val = 0

        self.env = game_env_instance

    def save_q_table(self, file_path: str) -> None:
        """
        Saves the agent's Q-table to a file using pickle.

        Args:
            file_path (str): The path to the file where the Q-table will be saved (e.g., "q_table.pkl").
        """
        try:
            # Open the file in binary write mode ('wb')
            with open(file_path, 'wb') as f:
                # Use pickle.dump() to write the q_table object to the file
                pickle.dump(self.q_table, f)
            print(f"Q-table successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, file_path: str) -> bool:
        """
        Loads the agent's Q-table from a file using pickle.

        Args:
            file_path (str): The path to the file from which the Q-table will be loaded.

        Returns:
            bool: True if the Q-table was loaded successfully, False otherwise.
        """
        try:
            # Check if the file exists before trying to load
            import os
            if not os.path.exists(file_path):
                print(f"No Q-table found at {file_path}. Starting with an empty Q-table.")
                self.q_table = {} # Ensure Q-table is empty if file doesn't exist
                return False

            # Open the file in binary read mode ('rb')
            with open(file_path, 'rb') as f:
                # Use pickle.load() to read the q_table object from the file
                self.q_table = pickle.load(f)
            print(f"Q-table successfully loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading Q-table from {file_path}: {e}")
            self.q_table = {} # In case of error, ensure Q-table is empty to avoid issues
            return False

    def get_q_value(self, state: np.ndarray[Tuple[int, int], np.dtype[np.int8]], action: Tuple[int, int]) -> float:
        hashable_state = self.make_state_immutable(state)
        if hashable_state not in self.q_table:
            self.q_table[hashable_state] = {}
        if action not in self.q_table[hashable_state]:
            self.q_table[hashable_state][action] = self.nothing_q_val # Default Q-value for new action in this state
        return self.q_table[hashable_state][action]


    def make_state_immutable(self, state: np.ndarray[Tuple[int, int], np.dtype[np.int8]]) -> Tuple[Tuple[int]]:
        return tuple(map(tuple, state))

    def update_q_table(self, 
                       state: np.ndarray[Tuple[int, int], np.dtype[np.int8]], 
                       action: Tuple[int, int],
                       next_state: np.ndarray[Tuple[int, int], np.dtype[np.int8]],
                       reward: float,
                       is_done: bool) -> None:
        """update the q table after taking an action from a state to a next state

        Args:
            state (_type_): current state
            action (_type_): current action taken from state
            next_state (_type_): next state after action completed
            reward (_type_): immediate reward obtainbed once in next_state
            is_done (bool): game is over or not
            env (_type_): _description_
        """
        hashable_state = self.make_state_immutable(state)
        #hashable_next_state = self.make_state_immutable(next_state)

        current_q = self.get_q_value(state, action)

        if is_done:
            max_future_q = self.nothing_q_val
        else:
            # Need to get valid actions from next_state to find max Q
            # This is where your agent might need a reference to the environment
            # or a way to query valid actions for a given state.
            #env.set_current_board_temporarily(next_state) # If env allows setting board
            valid_actions_from_next_state = self.env.get_valid_actions(state=next_state)
            #env.restore_original_board() # Restore if board was set temporarily

            if not valid_actions_from_next_state: # If no valid actions (e.g., full board, no winner)
                max_future_q = self.nothing_q_val
            else:
                q_values_for_next_state_actions = [
                    self.get_q_value(next_state, a_prime)
                    for a_prime in valid_actions_from_next_state
                ]
                max_future_q = max(q_values_for_next_state_actions)

        # Q-Learning update formula
        # Q(s,a)←Q(s,a)+α[R+γ max_a′​Q(s′,a′)−Q(s,a)]
        updated_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)

        # Store the updated Q-value
        if hashable_state not in self.q_table:
            self.q_table[hashable_state] = {}
        self.q_table[hashable_state][action] = updated_q

    def choose_action(self, current_state: np.ndarray[Tuple[int, int], np.dtype[np.int8]],
                      valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        hashable_state = self.make_state_immutable(current_state)
        if np.random.rand() < self.epsilon:
            # random choice
            N_valid_actions = len(valid_actions)
            choice = np.random.randint(low=0, high=N_valid_actions, size=1)
            return valid_actions[choice[0]]
        else:
            # exploit learned knowledge
            q_values = np.array([self.get_q_value(current_state, action) for action in valid_actions])
            ind_q_max = np.argmax(q_values)
            if type(ind_q_max) == np.int64:
                return valid_actions[ind_q_max]
            N_q_max = len(ind_q_max)
            if N_q_max > 1:
                ind_q_max = ind_q_max[np.random.randint(low=0, high=N_q_max)[0]]
            return valid_actions[ind_q_max]
            







if __name__=="__main__":
    # --- Initialize ---
    env = GameEnv()
    agent = Agent(game_env_instance=env) # Pass the env instance to the agent

    num_episodes = 10_000 # Play more games to learn well
    episode_rewards_history = [] # For plotting learning progress

    # Agent's epsilon decay settings
    agent.epsilon = 1.0
    agent.epsilon_decay_rate = 0.9999 # Slightly faster decay than 0.999, maybe 0.9995 or 0.999 for 100k episodes
    agent.final_epsilon = 0.01 # Allow it to exploit more at the end

    # --- Training Loop ---
    for episode in range(num_episodes):
        current_state_numpy = env.reset()
        total_episode_reward = 0.0
        is_done = False

        # Store the agent's last state-action pair for delayed reward update
        last_agent_state = None
        last_agent_action = None

        while not is_done:
            # Player 1 (Agent's) Turn
            if env.current_player == 1:
                valid_actions = env.get_valid_actions(state=current_state_numpy)
                if not valid_actions: # Handle unexpected no-valid-moves early
                    is_done = True
                    # If it's Player 1's turn and no moves, it's a draw if no winner yet
                    if env.winner is None:
                        # No update needed for the agent's last move, as it's a draw
                        pass
                    break

                chosen_action = agent.choose_action(current_state_numpy, valid_actions)
                
                # Store this state-action pair before taking the step
                last_agent_state = current_state_numpy
                last_agent_action = chosen_action

                next_state_numpy, reward, is_done = env.step(chosen_action)
                
                # Update agent's Q-table for *its own move*
                # This `reward` is for the action just taken by the agent.
                # If agent wins, reward is 1. If game continues, reward is 0.
                # If agent draws, reward is 0.
                agent.update_q_table(
                    state=last_agent_state,
                    action=last_agent_action,
                    reward=reward,
                    next_state=next_state_numpy,
                    is_done=is_done,
                )
                total_episode_reward += reward # Accumulate for episode tracking

                current_state_numpy = next_state_numpy # Advance state

                if is_done:
                    break # Game ended after agent's move

            # Player -1 (Opponent's) Turn - Self-play if episode is odd, Random if even
            elif env.current_player == -1:
                valid_actions = env.get_valid_actions(state=current_state_numpy)
                if not valid_actions: # Handle unexpected no-valid-moves early
                    is_done = True
                    # If it's Player -1's turn and no moves, it's a draw if no winner yet
                    if env.winner is None:
                        # Agent's last move led to a draw, which is already handled
                        pass
                    break

                if episode % 2 == 0: # Play against random opponent
                    chosen_action_opponent = env.random_model(valid_actions)
                else: # Play against self (agent's policy for opponent)
                    # When opponent is playing using agent's brain, their choices
                    # are also based on the same Q-table, but their 'rewards'
                    # are mirrored for the Q-table update, if you were to train them.
                    # Since we only update for Player 1, this is simpler.
                    chosen_action_opponent = agent.choose_action(current_state_numpy, valid_actions)

                next_state_numpy, reward, is_done = env.step(chosen_action_opponent)
                
                # IMPORTANT: If the opponent won, it means the agent's *last move* led to a loss.
                # We need to penalize that last agent's move.
                if is_done and env.winner == -1: # Opponent won
                    if last_agent_state is not None and last_agent_action is not None:
                        # Re-update the Q-value for the agent's last action with a negative reward
                        # The `max_future_q` for this re-update should be 0 since the game is over
                        agent.update_q_table(
                            state=last_agent_state,
                            action=last_agent_action,
                            reward=-1, # Agent lost
                            next_state=next_state_numpy, # This is the final state
                            is_done=True,
                        )
                    total_episode_reward = -1 # Final reward for Player 1 for this episode
                elif is_done and env.winner == 1: # Agent already won, should have exited earlier
                    pass # This case should ideally not be reached if previous checks are correct
                elif is_done and env.winner is None: # Draw
                    total_episode_reward = 0 # Final reward for Player 1 for this episode
                
                current_state_numpy = next_state_numpy # Advance state

                if is_done:
                    break # Game ended after opponent's move

        # Episode End: Decay Epsilon
        # Your epsilon decay is fine:
        agent.epsilon = max(agent.final_epsilon, agent.epsilon * agent.epsilon_decay_rate)
        
        episode_rewards_history.append(total_episode_reward)

        if episode % 1000 == 0:
            # Calculate average reward over the last 1000 episodes
            avg_reward_last_1000 = np.mean(episode_rewards_history[-1000:]) if episode > 0 else 0
            print(f"Episode {episode}: Avg reward over last 1000 episodes (P1 wins=1, P1 loss=-1, Draw=0): {avg_reward_last_1000:.3f}, epsilon = {agent.epsilon:.4f}")
            # Consider rendering the board more often if you want to inspect specific games
            # if episode % 10000 == 0:
            #     env.render()
            #     print(f"Final board for episode {episode}:")
            #     print(env.board)

    # After the training loop finishes
    agent.save_q_table("tic_tac_toe_q_table_6.pkl")
    print(f"Training finished. Q-table size: {len(agent.q_table)}")

    # --- Evaluation Phase (Optional, but highly recommended) ---
    print("\n--- Evaluation Phase (epsilon = 0) ---")
    agent.epsilon = 0.0 # Turn off exploration for evaluation
    num_eval_games = 1000
    agent_wins = 0
    opponent_wins = 0
    draws = 0

    for _ in range(num_eval_games):
        current_state = env.reset()
        is_done = False
        while not is_done:
            if env.current_player == 1:
                valid_actions = env.get_valid_actions(state=current_state)
                if not valid_actions: # Game ended (draw or opponent won previously)
                    is_done = True
                    break
                action = agent.choose_action(current_state, valid_actions)
                current_state, reward, is_done = env.step(action)
                if is_done and env.winner == 1:
                    agent_wins += 1
                elif is_done and env.winner is None:
                    draws += 1
                elif is_done and env.winner == -1:
                    opponent_wins += 1 # Should not happen if agent played optimally
            elif env.current_player == -1:
                valid_actions = env.get_valid_actions(state=current_state)
                if not valid_actions:
                    is_done = True
                    break
                # Opponent always plays optimally (using agent's learned policy, no exploration)
                # Or you can choose to evaluate against random for baseline comparison:
                # chosen_action_opponent = env.random_model(valid_actions)
                
                # For self-play evaluation, let the agent make the optimal move for Player -1 too
                chosen_action_opponent = agent.choose_action(current_state, valid_actions) 
                
                current_state, reward, is_done = env.step(chosen_action_opponent)
                if is_done and env.winner == -1:
                    opponent_wins += 1
                elif is_done and env.winner is None:
                    draws += 1
                elif is_done and env.winner == 1:
                    agent_wins += 1 # Should not happen if opponent played optimally
    
    print(f"Evaluation over {num_eval_games} games (Agent vs. Agent's Policy):")
    print(f"Agent Wins (Player 1): {agent_wins} ({agent_wins/num_eval_games*100:.2f}%)")
    print(f"Opponent Wins (Player -1): {opponent_wins} ({opponent_wins/num_eval_games*100:.2f}%)")
    print(f"Draws: {draws} ({draws/num_eval_games*100:.2f}%)")