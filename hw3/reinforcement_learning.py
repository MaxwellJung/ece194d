class Transition:
    def __call__(self, s, a):
        return None

class Environment:
    def _init__(self, initial_state, transition: Transition):
        self.state = initial_state
        self.transition = transition
    
    def process_action(self, a):
        self.state = self.transition(self.state, a)
        self.reward = 0
        
class Policy:
    def __call__(self, s):
        a = None
        return a
    
class Agent:
    def __init__(self, environment: Environment, policy: Policy):
        self.environment = environment
        self.policy = policy
        
    def choose_action(self):
        self.action = self.policy(self.environment.state)
        return self.action

class Simulation:
    def __init__(self, environment: Environment):
        '''Initializes one environment and one agent'''
        self.environment = environment
        self.agents: list[Agent] = [Agent(environment)]
        
    def add_agent(self):
        '''Add 1 more agent to the environment'''
        self.agents.append(Agent(self.environment))
        
    def play_all_agents(self):
        '''play each agent in the simulation once'''
        for agent in self.agents:
            agent.choose_action()
            self.environment.process_action(agent.action)