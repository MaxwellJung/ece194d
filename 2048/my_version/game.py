import numpy as np

def transition(board: np.ndarray, action):
    '''Take in a board_state and action and returns the next board_state'''
    
    # [TODO] code up game logic below
    next_board = board
    
    # generate random tile
    randx = np.random.randint(board.shape[0])
    randy = np.random.randint(board.shape[1])
    next_board[randx][randy] = 1
    
    return next_board