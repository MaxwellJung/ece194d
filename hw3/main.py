from reinforcement_learning import RLModel
from tetris import Tetris

def main():
    tetris_RL = RLModel(Tetris())
    tetris_RL.value_iteration()
    
if __name__ == '__main__':
    main()