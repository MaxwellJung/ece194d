import numpy as np
from game import TwntyFrtyEight
from agent import Agent

def main():
    g = TwntyFrtyEight()
    # g.play()
    
    agent = Agent(environ=g)
    # agent.w = np.load('w_star.npy')
    try:
        agent.find_optimal_weight(alpha=1e-8, tolerance=1e-5, alpha_decay=False)
    except KeyboardInterrupt:
        pass
    
    w_star = agent.w
    print(f'Optimal Weight: {w_star}')
    np.save('w_star.npy', w_star)
    print(agent.training_df)
    agent.training_df.to_csv('training_results.csv', index=False)
    print(f'Saved training results and current optimal weight')
    
if __name__ == '__main__':
    main()
