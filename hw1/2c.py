import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax 
import copy

arm_count = 10
class Environment:
    def __init__(self, mean) -> None:
        self.reward = None
        self.mean = mean
    
    def generate_reward(self, action):
        def pull_arm():
            return np.random.normal(loc=self.mean[action], scale=1)
        self.reward = pull_arm()
        
class Agent:
    def __init__(self) -> None:
        self.action = None
        self.t = 0
        self.total_reward = 0
        self.average_reward = None
        self.N = np.zeros(arm_count)
        self.Q = np.zeros(arm_count)
        self.H = np.zeros(arm_count)
        self.policy = None
        self.hyperparam = None
        
    def reset(self):
        self.action = None
        self.t = 0
        self.total_reward = 0
        self.average_reward = None
        self.N = np.zeros(arm_count)
        self.Q = np.zeros(arm_count)
        self.H = np.zeros(arm_count)

    def set_policy(self, policy, hyperparam):
        if policy == 'greedy':
            self.policy = self.greedy
        elif policy == 'epsilon-greedy':
            self.policy = self.epsilon_greedy
        elif policy == 'UCB':
            self.policy = self.ucb
        elif policy == 'gradient':
            self.policy = self.gradient
        else:
            self.policy = None
        self.hyperparam = hyperparam
        
    def greedy(self):
        if (self.t > self.hyperparam):
            return np.argmax(self.Q)
        else:
            return np.random.randint(arm_count)
        
    def epsilon_greedy(self):
        rng = np.random.uniform()
        if rng > self.hyperparam:
            return np.argmax(self.Q)
        else:
            return np.random.randint(arm_count)

    def ucb(self):
        return np.argmax(self.Q + self.hyperparam*np.sqrt(np.log(self.t+1)/(self.N+1)))

    def gradient(self):
        return np.random.choice(arm_count, p=softmax(self.H))
        
    def select_action(self):
        self.action = self.policy()
        self.N[self.action] += 1
        self.t += 1
        
    def accept_reward(self, reward):
        def update_H(a):
            if a == self.action:
                return self.H[a] + self.hyperparam*(reward-self.average_reward)*(1-softmax(self.H)[a])
            else:
                return self.H[a] - self.hyperparam*(reward-self.average_reward)*(softmax(self.H)[a])
        self.total_reward += reward
        self.average_reward = self.total_reward/self.t
        self.Q[self.action] = ((self.N[self.action]-1)*self.Q[self.action]+reward)/(self.N[self.action])
        self.H = np.vectorize(update_H)(np.arange(arm_count))
        
def simulate(mean, policy, hyperparam, time_horizon=1000):
    agent = Agent()
    env = Environment(mean)
    agent.set_policy(policy, hyperparam)
    regret_history = []
    for t in range(time_horizon):
        agent.select_action()
        env.generate_reward(action=agent.action)
        agent.accept_reward(reward=env.reward)
        regret = np.max(env.mean) - agent.average_reward
        regret_history.append(regret)
        
    return np.array(regret_history)

def main():
    sample_size = 30
    mean = np.random.normal(loc=0, scale=1, size=arm_count)
    
    
    print(f'Benchmarking Upper Confidence Bound')
    UCB_benchmark = np.mean(np.array([simulate(mean, 'UCB', 0.7) for i in range(sample_size)]), axis=0)
    print(f'Benchmarking Gradient')
    gradient_performances = np.mean(np.array([simulate(mean, 'gradient', 0.1) for i in range(sample_size)]), axis=0)
    
    X = np.arange(1000)
    
    plt.plot(X, UCB_benchmark, color='r', label='UCB')
    plt.plot(X, gradient_performances, color='g', label='gradient')
    
    plt.xlabel("Time (t)")
    plt.ylabel("Regret")
    plt.title("UCB vs gradient policy")
    plt.legend()
    
    plt.show()
    
if __name__ == '__main__':
    main()