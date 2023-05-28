from game import TwntyFrtyEight
from agent import Agent

def main():
    g = TwntyFrtyEight()
    # g.play()
    
    agent = Agent(environ=g)
    agent.policy_iteration()

if __name__ == '__main__':
    main()