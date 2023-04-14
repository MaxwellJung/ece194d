import numpy as np

arm_count = 10

class Environment:
    def __init__(self) -> None:
        self.reward = None
        self.mean = np.random.normal(loc=0, scale=1, size=arm_count)
    
    def generate_reward(self, action):
        def pull_arm():
            return np.random.normal(loc=self.mean[action], scale=1)
        self.reward = pull_arm()
class Agent:
    def __init__(self) -> None:
        self.t = 0
        self.total_reward = 0
        self.N = np.zeros(arm_count)
        self.Q = np.zeros(arm_count)
        self.policy = self.greedy
        self.action = None

    def set_policy(self, policy):
        self.policy = policy
        
    def greedy(self, N=5*arm_count):
        if (self.t > N):
            return np.argmax(self.Q)
        else:
            return np.random.randint(arm_count)
        
    def epsilon_greedy(self, epsilon=0.2):
        rng = np.random.uniform()
        if rng > epsilon:
            return np.argmax(self.Q)
        else:
            return np.random.randint(arm_count)

    def ucb(self, c=3):
        return np.argmax(self.Q + c*np.sqrt(np.log(self.t+1)/(self.N+1)))

    def gradient(self, a):
        return 0
        
    def select_action(self):
        self.action = self.policy()
        self.N[self.action] += 1
        self.t += 1
        
    def accept_reward(self, reward):
        self.total_reward += reward
        self.Q[self.action] = (self.N[self.action]-1)/(self.N[self.action])*self.Q[self.action] + (1/self.N[self.action])*reward

def main():
    env = Environment()
    bandit = Agent()
    bandit.set_policy(bandit.ucb)
    time_horizon = 1000
    
    for t in range(time_horizon):
        bandit.select_action()
        env.generate_reward(action=bandit.action)
        bandit.accept_reward(reward=env.reward)
    
    print(vars(bandit))
    print(vars(env))
    
if __name__ == '__main__':
    main()