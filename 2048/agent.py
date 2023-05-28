import numpy as np
from environment import Environment
from episode import Episode

class ValueFunction:
    def __init__(self, environ: Environment):
        self.environ = environ
        self.v_table = []
    
    def __call__(self, state: int, weight:np.ndarray=None):
        if self.environ.is_terminal_state(state): return 0
        if weight is None:
            return self.v_table[state]
        else:
            X = self.environ.get_feature_vector(state)
            return weight.dot(X)
    
class ActionValueFunction:
    def __init__(self, environ: Environment):
        self.environ = environ
        self.q_table = [[]]
        self.value = ValueFunction(environ)
    
    def __call__(self, state: int, action: int, weight:np.ndarray=None):
        if weight is None:
            return self.q_table[state][action]
        else:
            all_next_states = self.environ.get_all_next_states(state, action)
            rewards = [self.environ.reward(state, action, next_state) + self.value(next_state, weight) for next_state in all_next_states]
            
            return np.mean(rewards) # E(r+v(s')) when P(s') is uniform

class Agent:
    def __init__(self, environ: Environment):
        self.environ = environ
        self.q = ActionValueFunction(environ)
        self.w = None
        
    def policy_iteration(self, tolerance=1e-3):
        self.w = self.environ.rng.uniform(low=-1e2, high=1e2, size=len(self.environ.get_feature_vector(0)))
        while True:
            old_w = np.copy(self.w)
            policy = lambda s: self.greedy_policy(s)
            self.w = self.estimate_w(policy)
            if np.linalg.norm(old_w-self.w) < tolerance:
                break
        
    
    def estimate_w(self, policy, discount_factor=1, tolerance=1e-3):
        new_w = self.environ.rng.uniform(low=-1e2, high=1e2, size=len(self.environ.get_feature_vector(0)))
        update_count = 0
        episode_count = 0
        while True:
            epi = Episode(self.environ, policy)
            episode_count += 1
            old_w = np.copy(new_w)
            for t in range(epi.length):
                learning_rate = 1e-5 # alpha
                value = lambda state: self.q.value(state, weight=new_w)
                measurement = epi.rewardAt(t+1) + discount_factor*value(epi.stateAt(t+1)) # U_t
                estimate = value(epi.stateAt(t))
                grad = self.environ.get_feature_vector(epi.stateAt(t)) # gradient of (W^T)X is X
                update = learning_rate*(measurement - estimate)*grad
                new_w = new_w + update # w_t+1 = w_t + a[U_t-v(s_t, w_t)]*grad(v(s_t, w_t))
                update_count += 1
            
            # Print progress every 10 episodes
            if episode_count%1 == 0: print(f'{episode_count=} {update_count=} {new_w}')
            if np.linalg.norm(old_w-new_w) < tolerance: break
        return new_w
        
    def random_policy(self, s: int) -> int:
        valid_actions = self.environ.get_valid_actions(s)
        return valid_actions[self.environ.rng.integers(len(valid_actions))]
    
    def greedy_policy(self, s: int) -> int:
        valid_actions = self.environ.get_valid_actions(s)
        best_action = valid_actions[np.argmax([self.q(state=s, action=a, weight=self.w) for a in valid_actions])]
        return best_action
    
    def epsilon_greedy_policy(self, s: int, epsilon=0.1) -> int:
        valid_actions = self.environ.get_valid_actions(s)
        if self.environ.rng.ranom(1) < epsilon:
            return valid_actions[self.environ.rng.integers(len(valid_actions))]
        else:
            best_action = valid_actions[np.argmax([self.q(state=s, action=a, weight=self.w) for a in valid_actions])]
            return best_action