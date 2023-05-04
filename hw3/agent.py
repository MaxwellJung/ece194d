from environment import Environment

class Agent():
    def __init__(self, environment: Environment, policy):
        self.environment = environment
        self.policy = policy
        self.state_history = []
        self.action_history = []
        self.reward_history = []
    
    def choose_action(self):
        self.action_history.append(self.policy(self.environment.state))
        return self.action_history[-1]