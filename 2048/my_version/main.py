import numpy as np
import game

def main():
    board_shape = (4,4) # 4 x 4 board
    initial_board = np.zeros(board_shape)
    print(initial_board)
        
    while True:
        new_board = game.transition(initial_board, input('Enter action:'))
        print(new_board)
        
if __name__ == '__main__':
    main()