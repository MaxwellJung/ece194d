from reinforcement_learning import Simulation
from tetris import Tetris

def main():
    t = Tetris(starting_piece=0)
    t.start_interactive_play()
    
if __name__ == '__main__':
    main()