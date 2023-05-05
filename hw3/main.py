from reinforcement_learning import Simulation
from tetris import Tetris

def main():
    t = Tetris(starting_piece=0)
    while True:
        try:
            print(t.board)
            print(t.current_piece)
            t.place_piece(orientation=int(input('orientation:')), location=int(input('location:')))
        except Tetris.invalidMoveException:
            print(f'invalid move')
            continue
        except Tetris.gameOverException:
            print(f'game over')
            break
        else:
            t.current_piece = t.select_next_piece()

if __name__ == '__main__':
    main()