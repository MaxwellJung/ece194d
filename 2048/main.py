import numpy as np
from game import TwntyFrtyEight
from agent import Agent

rng = np.random.default_rng()

def main():
    g = TwntyFrtyEight()
    # g.play()
    
    agent = Agent(environ=g)
    agent.policy_iteration()

if __name__ == '__main__':
    main()