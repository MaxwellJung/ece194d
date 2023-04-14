import numpy as np
import copy

arm_count = 10

def softmax(x):
    return np.exp(x)/sum(np.exp(x))
class Environment:
    def __init__(self) -> None:
        self.reward = None
        self.mean = np.random.normal(loc=0, scale=1, size=arm_count)
        
    def reset(self):
        self.reward = None
    
    def generate_reward(self, action):
        def pull_arm():
            return np.random.normal(loc=self.mean[action], scale=1)
        self.reward = pull_arm()
class Agent:
    def __init__(self) -> None:
        self.policy = self.greedy
        self.action = None
        self.t = 0
        self.total_reward = 0
        self.average_reward = None
        self.N = np.zeros(arm_count)
        self.Q = np.zeros(arm_count)
        self.H = np.zeros(arm_count)
        self.hyperparam = None
        
    def reset(self):
        self.action = None
        self.t = 0
        self.total_reward = 0
        self.average_reward = None
        self.N = np.zeros(arm_count)
        self.Q = np.zeros(arm_count)
        self.H = np.zeros(arm_count)
        self.hyperparam = None

    def set_policy(self, policy, hyper_param):
        self.policy = policy
        self.hyperparam = hyper_param
        
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
        self.Q[self.action] = (self.N[self.action]-1)/(self.N[self.action])*self.Q[self.action] + (1/self.N[self.action])*reward
        self.H = np.vectorize(update_H)(np.arange(arm_count))
        
def simulate(agent: Agent, env: Environment, time_horizon=1000):
    for t in range(time_horizon):
        agent.select_action()
        env.generate_reward(action=agent.action)
        agent.accept_reward(reward=env.reward)
        
def evaluate(agent: Agent, env: Environment, policy, hyperparam):
    agent.set_policy(policy, hyperparam)
    simulate(agent, env)
    return np.max(env.mean)*agent.t - agent.total_reward

def test_epsilon_greedy(agent: Agent, env: Environment, e=0.1):
    agent.reset()
    env.reset()
    return evaluate(agent, env, agent.epsilon_greedy, hyperparam=e)

def test_gradient(agent: Agent, env: Environment, a=0.1):
    agent.reset()
    env.reset()
    return evaluate(agent, env, agent.gradient, hyperparam=a)
    
def main():
    sample_size = 30
    agent = Agent()
    env = Environment()
    
    for x in range(1, 100):
        hyperparam = e = x/100
        score = np.mean(np.fromfunction(lambda i: test_epsilon_greedy(copy.deepcopy(agent), copy.deepcopy(env), hyperparam), (sample_size,)))
        print(f'hyperparam: {hyperparam}, score: {score}')
        
if __name__ == '__main__':
    main()