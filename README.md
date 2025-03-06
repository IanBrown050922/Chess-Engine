# Chess-Engine
Chess Engine with Neural Network-Based Evaluation Function

The Board class in Board.py models chess boards and provides the ability to generate boards reachable from a given board via a single move.

The Model class in Model.py defines the convolutional neural network used as the evaluation function of the chess engine. The model outputs values from -1 to 1.

The Engine class in Engine.py defines the engine object, which includes a Model instance, the ability to train through self-play reinforcement learning, and the ability to play against the user.

The engine uses minimax search with alpha-beta pruning to search a certain number of moves ahead and choose the path leading to the best board-evaluation, assuming its opponent makes the best moves.

During training, the engine plays batches of games against itself, usually making the best moves, as determined by minimax and the current state of the model, but with a chance of random moves to encourage exploration. Once a game ends, all of the boards reached during that game are labeled with the outcome (1 for white win, -1 for black win, 0 for draw). After a batch of games are played, their boards are shuffled and the model is trained on them. The process repeats with the updated model.

chess_cnn.pth has model paramters obtained by very brief training (so the behavior is still inaccurate). The call to Engine.Train that obtained these paramters is commented-out in Main.py.
