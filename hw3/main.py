from reinforcement_learning import Simulation
from tetris import Tetris

def main():
    t = Tetris(starting_piece=1)
    try:
        t.place_piece(orientation=0, location=0)
        t.set_next_piece(3)
        t.place_piece(orientation=0, location=2)
        t.place_piece(orientation=1, location=2)
    except Tetris.invalidMoveException:
        print(f'invalid move')
    except Tetris.gameOverException:
        print(f'game over')

if __name__ == '__main__':
    main()