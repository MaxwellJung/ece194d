import logging
from collections import defaultdict
import numpy as np
from environment import Environment
from episode import Episode

# logging config
logging.basicConfig(level=logging.INFO, 
                    format="%(message)s",
                    handlers=[
                        logging.FileHandler("agent.log"),
                        logging.StreamHandler()],
                    )

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
        
    
    def estimate_w(self, policy, tolerance=1e-3, discount_factor=1, trace_decay_rate=1):
        '''
        Semi-gradient TD(lambda) for estimating v_hat close to v_pi
        algorithm from page 293 of Sutton Barto 2nd edition
        '''
        def show_progress():
            logging.info(f'{episode_count=} {update_count=} \n{w}')
            logging.info(f'{dict(stats)} win_rate={stats["win"]/episode_count:.2%} average_steps={update_count/episode_count:.2f}')
            
        stats = defaultdict(int)
        
        w = self.w
        update_count = 0
        episode_count = 0
        while True:
            epi = Episode(self.environ, policy)
            episode_count += 1
            z = 0
            old_w = np.copy(w)
            for t in range(epi.length):
                learning_rate = 1e-6 # alpha
                value = lambda state: self.q.value(state, weight=w)
                measurement = epi.rewardAt(t+1) + discount_factor*value(epi.stateAt(t+1)) # U_t
                estimate = value(epi.stateAt(t))
                grad = self.environ.get_feature_vector(epi.stateAt(t)) # gradient of (W^T)X is X
                z = discount_factor*trace_decay_rate*z + grad
                update = learning_rate*(measurement - estimate)*z
                w = w + update # w_t+1 = w_t + a[U_t-v(s_t, w_t)]*grad(v(s_t, w_t))
                update_count += 1
            
            # Record episode stats
            stats[self.environ.get_state_status(epi.state_history[-1])] += 1
            
            # Print progress every 100 episodes
            if episode_count%100 == 0: show_progress()
            if np.linalg.norm(old_w-w) < tolerance: break
            
        logging.info(f'------------------------Final convergence------------------------')
        show_progress()
        logging.info(f'-----------------------------------------------------------------')
        return w
        
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