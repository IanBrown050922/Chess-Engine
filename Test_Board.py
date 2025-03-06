from Board import Board
import random
from itertools import islice



'''
Testing Board class functionality
'''



# Plays random game up to max_moves many moves
def play_random_game(max_moves):
    # Start with the initial board.
    board = Board()  # Assumes __init__ sets up the standard starting position.
    print("Initial Board:")
    print(board)
    
    for move_number in range(1, max_moves + 1):
        # Generate a list of all possible moves from the current board.
        possible_moves = list(board.generate_next_boards())
        
        # If no moves are available, the game is over.
        if not possible_moves:
            print("No moves available. Game Over.")
            break
        
        # Randomly select one of the possible moves.
        board = random.choice(possible_moves)
        
        # Print the board after the move.
        print(f"\nAfter move {move_number}:")
        print(board)


# Plays game with moves indexed in moves
def play_moves(moves, print_options=False):
    board = Board()
    print('Board 0:')
    print(board)

    for i, m in enumerate(moves):
        next_board = next(islice(board.generate_next_boards(), m, m+1))
        print(f'\nMove {i+1}:')
        print(next_board)
        board = next_board
    
    if print_options:
        for i, next_board in enumerate(board.generate_next_boards()):
            print(f'\nOption {i}:')
            print(next_board)


if __name__ == '__main__':
    # ex. play to get white pawn promoted to queen
    play_moves([0, 0, 0, 0, 12, 20, 0, 21, 11, 0, 11, 0, 14], True)

    # ex. play random game to 200 moves (3-fold repetition draw not enforced here)
    # play_random_game(200)