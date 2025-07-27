import numpy as np
from GameEnv import GameEnv, Agent



# Conceptual Outer Game Loop
env = GameEnv()
agent = Agent(game_env_instance=env) # Pass the env instance to the agent
q_table_file = "tic_tac_toe_q_table_6.pkl" # Make sure this matches your save filename
agent.load_q_table(q_table_file)
agent.epsilon = 0.0 # Agent should now only exploit its learned knowledge
while True: # Loop indefinitely until user decides to quit
    # 1. Reset the game environment for a fresh start
    current_state_numpy = env.reset()
    is_done = False
    
    # Optional: Decide who goes first
    # You can randomly choose, or always let human/AI start.
    # If human plays as Player 1 (value 1), AI plays as Player 2 (value -1).
    # current_player_turn = random_choice_between_human_or_ai_starts

    print("\n--- New Game Started! ---")

    # 2. Inner Game Loop (Play a single game/episode)
    while not is_done:
        # A. Render the board
        # This is CRUCIAL for human play. Make sure your env.render() is good!
        env.render() # Displays the current board state

        # B. Determine whose turn it is
        # (This is implicitly handled by env.current_player)
        if env.current_player == -1: # Assuming Human is Player 1
            print("\nYour turn (Player X).")
            # --- Human's Turn Logic ---
            valid_moves = env.get_valid_actions(state=current_state_numpy)
            human_action = None
            action_is_valid = False

            while not action_is_valid:
                # 1. Prompt for input
                # Example: "Enter your move (row,col, e.g., 0,0 for top-left): "
                user_input_string = input("Your move (row,col): ")

                # 2. Parse and Validate Human Input
                # Try to convert input string to a (row, col) tuple (e.g., "0,0" -> (0,0))
                # Check:
                #   - Is it in the correct format? (e.g., two numbers separated by comma)
                #   - Are numbers within board bounds (0 to N_rows-1)?
                #   - Is the chosen cell actually in the 'valid_moves' list?
                # If any check fails, print an error message and loop again to prompt.
                
                try:
                    # (Conceptual parsing: split string by comma, convert to int, create tuple)
                    row_str, col_str = user_input_string.split(',')
                    human_action = (int(row_str.strip()), int(col_str.strip()))

                    if human_action in valid_moves:
                        action_is_valid = True
                    else:
                        print("Invalid move: Cell is not empty or out of bounds. Try again.")
                except ValueError:
                    print("Invalid format. Please enter as 'row,col' (e.g., '0,0').")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}. Please try again.")

            # 3. Human takes action
            # The step method will handle changing board, checking win, switching player
            next_state_numpy, reward_human, is_done = env.step(human_action)
            current_state_numpy = next_state_numpy # Update state for next iteration

        else: # AI's Turn (Player -1)
            print("\nAI's turn (Player O)...")
            # --- AI's Turn Logic ---
            valid_moves = env.get_valid_actions(state=current_state_numpy)
            if not valid_moves: # Should ideally be caught by env.is_game_over earlier for draws
                print("No valid moves for AI. This should indicate a draw or pre-existing end condition.")
                is_done = True
                break

            # AI chooses action based on its Q-table (pure exploitation as epsilon=0)
            ai_action = agent.choose_action(current_state_numpy, valid_moves)
            print(f"AI chooses: {ai_action[0]},{ai_action[1]}")

            # AI takes action
            next_state_numpy, reward_ai, is_done = env.step(ai_action)
            current_state_numpy = next_state_numpy # Update state for next iteration

        # After each step (human or AI), check if the game is over
        if is_done:
            env.render() # Render final board
            break # Exit inner game loop

    # 3. Game Over Announcements
    print("\n--- Game Over! ---")
    if env.winner is not None:
        if env.winner == 1:
            print("Congratulations! You (Player X) won!")
        else: # env.winner == -1
            print("AI (Player O) won! Better luck next time!")
    else: # env.winner is None and is_done is True, implies a draw
        print("It's a draw!")

    # 4. Ask to Play Again
    play_again = input("\nPlay another game? (yes/no): ").lower()
    if play_again != 'yes':
        print("Thanks for playing! Goodbye.")
        break # Exit outer game loop
