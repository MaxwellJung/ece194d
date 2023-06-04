import numpy as np
from game import TwntyFrtyEight
from agent import Agent

def main():
    g = TwntyFrtyEight()
    # g.play()
    
    agent = Agent(environ=g)
    agent.w = np.load('w_star.npy')
    try:
        agent.find_optimal_weight(alpha=1e-8, tolerance=0, alpha_decay=False)
    except KeyboardInterrupt:
        pass
    
    w_star = agent.w
    print(f'Optimal Weight: {w_star}')
    np.save('w_star.npy', w_star)

if __name__ == '__main__':
    main()
