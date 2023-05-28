import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

starting_state = 500
losing_state = 0
winning_state = 1001

def main():
    v_true = dp()
    v_sa = state_aggregation(max_episodes=5000)
    v_pb = polynomial_basis(max_episodes=5000)
    v_fb = fourier_basis(max_episodes=5000)
    
    fig, ax = plt.subplots()
    ax.plot(range(len(v_true)), v_true, color='r', label='True Values')
    ax.plot(range(len(v_sa)), v_sa, color='g', label='State Aggregation')
    ax.plot(range(len(v_pb)), v_pb, color='b', label='Polynomial Basis')
    ax.plot(range(len(v_fb)), v_fb, color='y', label='Fourier Basis')

    ax.set(xlabel='State', ylabel='Value', title='Value Functions')
    ax.grid()
    plt.legend()
    plt.show()

class Episode:
    def __init__(self):
        s = starting_state
        self.state_history = [s]
        self.reward_history = [None]
        
        while not (s == losing_state or s == winning_state):
            s_prime, reward = transition(s)
            self.state_history.append(s_prime)
            self.reward_history.append(reward)
            s = s_prime
            
    def state_at(self, t):
        if 0 <= t < len(self.state_history): return self.state_history[t]
        else: return None
        
    def reward_at(self, t):
        if 0 <= t < len(self.reward_history): return self.reward_history[t]
        else: return None
    
    
def dp():
    values = np.zeros(1002)
    
    while True:
        old_values = values.copy()
        for s in range(1,1001):
            expected_value = 0
            for s_prime in range(s-100, s+100+1):
                if s_prime == s: continue
                if s_prime < losing_state: s_prime = losing_state
                elif s_prime > winning_state: s_prime = winning_state
                reward = 0
                if s_prime == winning_state: reward = 1
                elif s_prime == losing_state: reward = -1
                expected_value += 1/200*(reward+values[s_prime])
            values[s] = expected_value
        
        if np.linalg.norm(old_values-values) < 0.001:
            break
    
    return values

def state_aggregation(max_episodes=5000):
    w = np.zeros(10)
    discount = 1
    alpha = 2e-5
    
    for i in range(max_episodes):
        epi = Episode()
        g_t = 0
        old_w = w.copy()
        for t in reversed(range(len(epi.state_history)-1)):
            s_t = epi.state_at(t)
            group_index = (s_t-1)//100
            g_t = epi.reward_at(t+1) + discount*g_t
            estimate = w[group_index]
            grad = np.zeros(10)
            grad[group_index] = 1
            w = w + alpha*(g_t-estimate)*grad
            
        if i%1000 == 0: print(f'{i=} {w}')
        if np.linalg.norm(old_w-w) < 0.000001:
            break
    
    values = np.zeros(1002)
    for s in range(len(values)):
        if s == losing_state or s == winning_state:
            continue
        group_index = (s-1)//100
        values[s] = w[group_index]
        
    return values

def get_polynomial_feature(state):
    x = np.arange(6)
    x = np.power(state/1000, x)
    return x

def polynomial_basis(max_episodes=5000):
    w = np.zeros(6)
    discount = 1
    alpha = 1e-4
    
    for i in range(max_episodes):
        epi = Episode()
        g_t = 0
        old_w = w.copy()
        for t in reversed(range(len(epi.state_history)-1)):
            s_t = epi.state_at(t)
            g_t = epi.reward_at(t+1) + discount*g_t
            x = get_polynomial_feature(s_t)
            estimate = w.dot(x)
            grad = x
            w = w + alpha*(g_t-estimate)*grad
            
        if i%1000 == 0: print(f'{i=} {w}')
        if np.linalg.norm(old_w-w) < 0.000001:
            break
    
    values = np.zeros(1002)
    for s in range(len(values)):
        if s == losing_state or s == winning_state:
            continue
        x = get_polynomial_feature(s)
        values[s] = w.dot(x)
        
    return values

def get_fourier_feature(state):
    c = np.arange(6)
    x = np.cos(np.pi*c*state/1000)
    return x

def fourier_basis(max_episodes=5000):
    w = np.zeros(6)
    discount = 1
    alpha = 5e-5
    
    for i in range(max_episodes):
        epi = Episode()
        g_t = 0
        old_w = w.copy()
        for t in reversed(range(len(epi.state_history)-1)):
            s_t = epi.state_at(t)
            g_t = epi.reward_at(t+1) + discount*g_t
            x = get_fourier_feature(s_t)
            estimate = w.dot(x)
            grad = x
            w = w + alpha*(g_t-estimate)*grad
            
        if i%1000 == 0: print(f'{i=} {w}')
        if np.linalg.norm(old_w-w) < 0.000001:
            break
    
    values = np.zeros(1002)
    for s in range(len(values)):
        if s == losing_state or s == winning_state:
            continue
        x = get_fourier_feature(s)
        values[s] = w.dot(x)
        
    return values
        
def transition(state):
    magnitude = rng.integers(1,101)
    direction = 1 if rng.random(1) < 0.5 else -1
    
    next_state = state + direction*magnitude
    
    if next_state < losing_state: next_state = losing_state
    elif next_state > winning_state: next_state = winning_state
    
    reward = 0
    if next_state == winning_state: reward = 1
    elif next_state == losing_state: reward = -1
    
    return next_state, reward

if __name__ == '__main__':
    main()