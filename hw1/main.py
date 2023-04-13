import numpy as np

arm_count = 10

class Environment:
    def __init__(self) -> None:
        self.state = None
        self.reward = 0
        self.mean = np.random.normal(loc=0, scale=1, size=arm_count)
    
    def process_action(self, action):
        def pull_arm():
            return np.random.normal(loc=self.mean[action], scale=1)
        self.state = self.state
        self.reward = pull_arm()

class Agent:
    def __init__(self) -> None:
        self.Q = np.zeros(arm_count)
        self.action = 0
        self.t = 0
        self.N = 100
        
    def select_action(self, state, reward):
        def greedy():
            if (self.t > self.N):
                self.action = np.argmax(self.Q)
            else:
                self.action = np.random.randint(arm_count)
        
        def e_greedy(e):
            pass
        
        def ucb(c):
            pass
        
        def gradient(a):
            pass
        
        self.Q[self.action] += reward
        greedy()
        self.t += 1


def main():
    env = Environment()
    bandit = Agent()
    time_horizon = 1000
    
    for t in range(time_horizon):
        bandit.select_action(state=env.state, reward=env.reward)
        env.process_action(action=bandit.action)
    
    print(bandit.Q)
    print(env.mean)
if __name__ == '__main__':
    main()