class MDP():
    def _init__(self):
        self.state = None
        self.reward = 0
    
    def transition(self, action) -> None:
        self.state = None
        self.reward = 0