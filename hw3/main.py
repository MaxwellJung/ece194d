from reinforcement_learning import RLModel, State
from tetris import Tetris
import numpy as np

def main():
    t = Tetris()
    tetris_RL = RLModel(t)
    vals = tetris_RL.value_iteration()
    
    
if __name__ == '__main__':
    main()