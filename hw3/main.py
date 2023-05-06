from reinforcement_learning import RLModel
from tetris import Tetris

def main():
    tetris_RL = RLModel(Tetris())
    episode = tetris_RL.generate_episode()
    
    print(episode)
    
if __name__ == '__main__':
    main()