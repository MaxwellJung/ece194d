from abc import ABC, abstractmethod
import random

class State:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def __repr__(self) -> str:
        return f'State = {self.kwargs}'
    
    def __eq__(self, other) -> bool:
        return repr(self) == repr(other)
    
    def __hash__(self) -> int:
        return hash(str(self))

class Action:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def __repr__(self) -> str:
        return f'Action = {self.kwargs}'
    
    def __eq__(self, other) -> bool:
        return self.kwargs == other.kwargs
    
    def __hash__(self) -> int:
        return hash(str(self))

class Environment(ABC): 
    def _init__(self):
        self.state: State = None
        self.reward: float = 0
        self.all_states: list[State] = []
        self.all_actions: list[Action] = []
        
    @abstractmethod
    def set_state(self, s: State):
        pass
        
    @abstractmethod
    def get_state(self):
        pass
    
    @abstractmethod
    def get_reward(self):
        pass
    
    @abstractmethod
    def transition(self, a: Action):
        pass
    
    def __repr__(self) -> str:
        return f'{self.state=}, {self.reward=}'
        
class Agent(ABC):
    def __init__(self, environment: Environment):
        self.environment: Environment = environment
    
    def policy(self) -> Action:
        return random.choice(self.environment.all_actions)

class RLModel:
    def __init__(self, environment: Environment):
        '''Initializes one environment and one agent'''
        self.environment = environment
        self.agents: list[Agent] = [Agent(environment)]
        
    def add_agent(self):
        '''Add 1 more agent to the environment'''
        self.agents.append(Agent(self.environment))
        
    def generate_episode(self):
        '''play each agent in the simulation until terminal state'''
        episode = []
        while True:
            for agent in self.agents:
                episode.append(self.environment.get_state())
                a = agent.policy()
                episode.append(a)
                self.environment.transition(a)
                episode.append(self.environment.get_reward())
                if self.environment.get_state() == State(terminal=True):
                    return episode
                
    def value_iteration(self):
        values = {a: 0 for a in self.environment.all_states}
        values[State(terminal=True)] = 0
        
        discount_factor = 1
        
        for i in range(10):
            for s in self.environment.all_states:
                potential_values = []
                for a in self.environment.all_actions:
                    self.environment.set_state(s)
                    self.environment.transition(a)
                    next_state = self.environment.get_state()
                    r = self.environment.get_reward()
                    potential_values.append(r+discount_factor*values[next_state])
                values[s] = max(potential_values)
            print(values.values())
            
        return values