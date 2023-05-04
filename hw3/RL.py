from mdp import MDP
from agent import Agent

class RL:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        self.agents: list[Agent] = []
        
    def add_agent(self):
        self.agents.append(Agent(self.environment))
        
    def play_all_agents(self):
        for agent in self.agents:
            self.environment.process_action(agent.choose_action())