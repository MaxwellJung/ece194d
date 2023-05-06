from abc import ABC, abstractmethod
import random

class State:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def __repr__(self) -> str:
        return f'State = {self.kwargs}'
    
    def __eq__(self, other) -> bool:
        return self.kwargs == other.kwargs

class Action:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def __repr__(self) -> str:
        return f'Action = {self.kwargs}'
    
    def __eq__(self, other) -> bool:
        return self.kwargs == other.kwargs

class Environment(ABC): 
    def _init__(self):
        self.state: State = None
        self.reward: float = 0
        self.all_actions: list[Action] = []
        
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