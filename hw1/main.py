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
        self.policy = greedy
        self.Q = np.zeros(arm_count)
        self.action = 0
        self.t = 0
        
    def set_policy(self, name):
        if name == 'greedy':
            self.policy = greedy
        elif name == 'epsilon_greedy':
            self.policy = epsilon_greedy
        elif name == 'ucb':
            self.policy = ucb
        elif name == 'gradient':
            self.policy = gradient
        else:
            self.policy = greedy
        
    def select_action(self, state, reward):
        self.Q[self.action] += reward
        self.action = self.policy(t=self.t, Q=self.Q)
        self.t += 1

def greedy(t, Q, N=100):
    if (t > N):
        return np.argmax(Q)
    else:
        return np.random.randint(arm_count)
    
def epsilon_greedy(t, Q, epsilon=0.2, N=None):
    rng = np.random.uniform()
    if rng > epsilon:
        return np.argmax(Q)
    else:
        return np.random.randint(arm_count)

def ucb(c):
    return 0

def gradient(a):
    return 0

def main():
    env = Environment()
    bandit = Agent()
    bandit.set_policy('epsilon_greedy')
    time_horizon = 1000
    
    for t in range(time_horizon):
        bandit.select_action(state=env.state, reward=env.reward)
        env.process_action(action=bandit.action)
    
    print(bandit.Q)
    print(env.mean)
    
if __name__ == '__main__':
    main()