import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from Board import Board
from Model import ChessCNN


class Engine:
    def __init__(self, model_path=None, device='cpu'):
        self.device = device

        # Create NN and load trained parameters if available
        self.model = ChessCNN()
        if model_path:
            self.load_model(model_path)

        self.model.to(self.device)


    # Save trained parameters
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    
    # Load trained parameters
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    

    # Return NN's raw evaluation of board
    def evaluate_board(self, board):
        self.model.eval() # evaluation mode (disables dropout, etc.)
        x = board.board.unsqueeze(0).to(self.device) # (12, 8, 8) --> add batch dimension (1, 12, 8, 8)
        with torch.no_grad():
            value = self.model(x)
        return value.item()


    # Return best board reachable in depth many plys, assuming opponent makes best moves
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        # termination criteria
        next_boards = list(board.generate_next_boards())
        if not next_boards: # if no moves are available
            return board.get_result(), None # this is a terminal board, so get the actual value of the position
        elif depth == 0: # depth limit reached
            return self.evaluate_board(board), None # use NN evaluation

        best_move = None
        
        if maximizing_player:
            max_eval = float('-inf')
            for next_board in next_boards:
                eval_val, _ = self.minimax(next_board, depth - 1, alpha, beta, False)
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = next_board
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for next_board in next_boards:
                eval_val, _ = self.minimax(next_board, depth - 1, alpha, beta, True)
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = next_board
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval, best_move


    # Return next board
    def select_move(self, board, search_depth, epsilon=0.0):
        # epsilon chance of random move (for exploration in training)
        if random.random() < epsilon:
            return random.choice(list(board.generate_next_boards()))
        else:
            return self.minimax(board, search_depth, float('-inf'), float('inf'), board.white_to_move)[1]


    # Play a single game via self-play, using the current state of the engine/NN to make apparent best moves;
    # to avoid excessive dictionary copying, we enforce 3-fold repetion rule here, rather than in the board class
    def self_play_game(self, search_depth, epsilon):
        boards = []
        board = Board() # initial board
        board_counts = {} # track number of times each position occurs in case of 3-fold repetition = draw
        
        # Loop until no moves remain
        i = 1
        while any(board.generate_next_boards()):
            # hashable representation of board state
            hashable_board = hash(tuple(board.board.flatten().tolist()))
            if hashable_board in board_counts:
                board_counts[hashable_board] += 1
            else:
                board_counts[hashable_board] = 1

            # detect 3-fold repetition
            if board_counts[hashable_board] >= 3:
                break

            boards.append(board.board.clone()) # board tensors
            board = self.select_move(board, search_depth, epsilon)
            print()
            print(f'Move {i}')
            print(board)
            i += 1
        
        # fixed perspective: 1 = white win, -1 = black win, 0 = draw
        result = board.get_result()
        return [(board, result) for board in boards] # tensors associated with result score
    

    # Train the engine's NN through self-play reinforcement learning
    def train(self, search_depth=2, epsilon=0.1, num_game_batches=10, games_per_batch=10, epochs_per_batch=1, batch_size=32, lr=0.001, wd=1e-4):
        self.model.train() # training mode

        # define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        loss = nn.MSELoss()

        # Generate num_game_batches many batches of games.
        # Each batch of games has batch_size_games many games.
        # Each batch of games are generated through self play using minimax, where position evaluations are
        # computed using the NN in it's current state.
        # The NN is then trained/updated on the board states reached during the games
        # before the next batch of games is generated.
        for _ in range(num_game_batches):  # repeat until all games are generated
            training_dataset = []

            # generate a batch of self-play games
            print(f'Generating batch of {games_per_batch} games')
            for i in range(games_per_batch):
                print(f'Game {i+1}')
                training_dataset.extend(self.self_play_game(search_depth, epsilon))

            # convert dataset into tensors
            boards = torch.stack([data[0] for data in training_dataset])  # (N, 12, 8, 8)
            targets = torch.tensor([data[1] for data in training_dataset], dtype=torch.float32).unsqueeze(1)  # (N, 1)
            boards = boards.to(self.device)
            targets = targets.to(self.device)

            # label each board with the outcome of the game it belonged to: 1, 0, or -1
            dataset = TensorDataset(boards, targets)
            # create DataLoader for mini-batch training
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # train on the new batch of games
            print('updating')
            for _ in range(epochs_per_batch):
                for batch_boards, batch_targets in dataloader:
                    optimizer.zero_grad() # clear previous gradients
                    predictions = self.model(batch_boards) # forward pass
                    #print(predictions)
                    #print(batch_targets)
                    output = loss(predictions, batch_targets) # compute loss
                    print(f'loss: {output.item():.5f}')
                    output.backward() # backpropagation
                    optimizer.step() # update weights
    

    # Play game against user
    def play_against_user(self):
        board = Board()

        while True:
            print('\nCurrent Board:')
            print(board)

            move_options = list(board.generate_next_boards())

            if not move_options:
                print('Game over!')
                break

            if board.white_to_move:
                print('\nYour Move (White):')

                for i, move in enumerate(move_options):
                    print(f'\n{i}:')
                    print(move)

                while True:
                    try:
                        choice = int(input('\nEnter move choice: '))
                        if 0 <= choice < len(move_options):
                            board = move_options[choice]
                            break
                        else:
                            print('Invalid choice. Enter a number from the list.')
                    except ValueError:
                        print('Invalid input. Enter a number.')
            else:
                print('\nEngine\'s Move (Black):')
                board = self.select_move(board, 2)