import numpy as np
import matplotlib.pyplot as plt

world = [[-1]*12]*3
cliff = [-100]*10
cliff.insert(0, -1)
cliff.append(-1)
world.append(cliff)
world = np.array(world)

def state_to_position(state): return np.unravel_index(state, world.shape)
def position_to_state(pos): return np.ravel_multi_index(pos, world.shape)

start_state = position_to_state((3,0))
end_state = position_to_state((3,11))

class InvalidMoveException(Exception): pass

def move_down(state):
    current_pos = state_to_position(state)
    if 0 <= current_pos[0]+1 <= 3:
        next_pos = (current_pos[0]+1, current_pos[1])
    else:
        raise InvalidMoveException
    
    return position_to_state(next_pos)

def move_left(state):
    current_pos = state_to_position(state)
    if 0 <= current_pos[1]-1 <= 11:
        next_pos = (current_pos[0], current_pos[1]-1)
    else:
        raise InvalidMoveException
    
    return position_to_state(next_pos)

def move_up(state):
    current_pos = state_to_position(state)
    if 0 <= current_pos[0]-1 <= 3:
        next_pos = (current_pos[0]-1, current_pos[1])
    else:
        raise InvalidMoveException
    
    return position_to_state(next_pos)

def move_right(state):
    current_pos = state_to_position(state)
    if 0 <= current_pos[1]+1 <= 11:
        next_pos = (current_pos[0], current_pos[1]+1)
    else:
        raise InvalidMoveException
    
    return position_to_state(next_pos)
    
actions = {0: move_right,
           1: move_up,
           2: move_left,
           3: move_down,}

def transition(state, action):
    next_state = actions[action](state)
    reward = -1
    
    if world[state_to_position(next_state)] == -100:
        reward = -100
        next_state = start_state
    
    return next_state, reward

def get_possible_actions(state):
    possible_actions = []
    for a, action_logic in actions.items():
        try: next_state = action_logic(state)
        except InvalidMoveException: continue
        else: possible_actions.append(a)
        
    return possible_actions

def greedy_policy(state, q_table):
    action_values = [q_table[state][action] for action in get_possible_actions(state)]
    return actions[np.argmax(action_values)]
        
    return epsilon_greedy(0.1)

def epsilon_greedy_policy(state, q_table, episilon=0.1):
    actions = get_possible_actions(state)
    if np.random.rand(1) < episilon:
        return actions[np.random.randint(len(actions))]
    else:
        action_values = [q_table[state][action] for action in actions]
        return actions[np.argmax(action_values)]
    

def q_learning(alpha, max_episodes=1000):
    q_table = np.zeros((np.prod(world.shape), len(actions)))
    total_reward_history = []
    
    for episode_count in range(max_episodes):
        total_reward = 0
        s = start_state
        while s != end_state:
            a = epsilon_greedy_policy(s, q_table, episilon=0.1)
            s_prime, r = transition(s, a)
            measurement = r + np.max([q_table[s_prime][a_prime] for a_prime in get_possible_actions(s_prime)])
            estimate = q_table[s][a]
            q_table[s][a] = q_table[s][a] + alpha*(measurement - estimate)
            s = s_prime
            total_reward += r
        total_reward_history.append(total_reward)
            
    return q_table, total_reward_history

def sarsa(alpha, max_episodes=1000):
    q_table = np.zeros((np.prod(world.shape), len(actions)))
    total_reward_history = []
    
    for episode_count in range(max_episodes):
        total_reward = 0
        s = start_state
        while s != end_state:
            a = epsilon_greedy_policy(s, q_table, episilon=0.1)
            s_prime, r = transition(s, a)
            a_prime = epsilon_greedy_policy(s_prime, q_table, episilon=0.1)
            measurement = r + q_table[s_prime][a_prime]
            estimate = q_table[s][a]
            q_table[s][a] = q_table[s][a] + alpha*(measurement - estimate)
            s = s_prime
            total_reward += r
        total_reward_history.append(total_reward)
            
    return q_table, total_reward_history
    

def main():
    sarsa_q_table, sarsa_reward_trend = sarsa(alpha=0.05, max_episodes=5000)
    q_learning_q_table, q_learning_reward_trend = q_learning(alpha=0.05, max_episodes=5000)
    
    samples = 10
    
    smooth_sarsa_reward_trend = np.average(np.array(sarsa_reward_trend).reshape(-1, samples), axis=1)
    smooth_q_learning_reward_trend = np.average(np.array(q_learning_reward_trend).reshape(-1, samples), axis=1)
    
    fig, ax = plt.subplots()
    ax.plot(range(1,len(smooth_sarsa_reward_trend)+1), smooth_sarsa_reward_trend, color='b', label='Sarsa')
    ax.plot(range(1,len(smooth_q_learning_reward_trend)+1), smooth_q_learning_reward_trend, color='r', label='Q-learning')

    ax.set(xlabel='Episodes', ylabel='Sum of rewards', title='Performance')
    ax.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()