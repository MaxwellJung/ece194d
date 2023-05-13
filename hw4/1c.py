import numpy as np
import matplotlib.pyplot as plt

def generate_episode(policy: np.ndarray, p_h=0.55, max_steps=1000):
    state_history = []
    action_history = []
    reward_history = []
    
    initial_state = np.random.randint(100)
    state_history.append(initial_state)
    
    for i in range(max_steps):
        state = int(state_history[-1])
        action = policy[state]
        action_history.append(action)
        
        heads = np.random.random() < p_h
        next_state = state + action if heads else state - action
        state_history.append(next_state)
        
        reward = 1 if next_state >= 100 else 0
        reward_history.append(reward)
        
        if next_state <= 0 or next_state >= 100:
            break
        
    state_history.pop()
    
    return state_history, action_history, reward_history

def monte_carlo(policy: np.ndarray, p_h, num_of_episodes=1000, discount=1):
    table = {s: [] for s in range(101)}
    
    for i in range(num_of_episodes):
        state_history, action_history, reward_history = generate_episode(policy, p_h=p_h)
        g = 0
        for t, s in reversed(list(enumerate(state_history))):
            g = reward_history[t] + discount*g
            table[s].append(g)
            
    return np.array([np.mean(table[s]) if len(table[s]) > 0 else 0 for s in table])
    
def main():
    p_h = 0.25
    values = monte_carlo(np.load(f'{p_h} policy.npy'), p_h=p_h, num_of_episodes=10000)
    np.save(f'{p_h} monte_carlo', values)
    
    expected = np.load(f'{p_h} values.npy')
    actual = values
    difference = np.linalg.norm(expected-actual)
    print(f'{expected=}\n{actual=}\n{difference=}')

if __name__ == '__main__':
    main()