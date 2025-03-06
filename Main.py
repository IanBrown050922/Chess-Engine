from Engine import Engine

if __name__ == '__main__':
    engine = Engine(device='cuda')

    # train and save parameters
    # engine.train(search_depth=2, epsilon=0.2, num_game_batches=5, games_per_batch=20, epochs_per_batch=1, batch_size=32, lr=1e-3, wd=1e-4)
    # engine.save_model('chess_cnn.pth')

    # load parameters
    engine.load_model('chess_cnn.pth')

    # play against user
    engine.play_against_user()