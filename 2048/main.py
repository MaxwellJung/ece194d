from typing import Any
import numpy as np
import logic
import math

rng = np.random.default_rng()

def main():
    policy_iteration()

def policy_iteration():
    w = rng.uniform(low=-1e2, high=1e2, size=3)
    starting_w = w.copy()
    for i in range(100):
        q = actionValueFunction(w)
        pi = Policy(q)
        w = sgd(policy=pi)
    print(starting_w)
    print(w)
    
def sgd(policy, tolerance=1e-2, episode_length=1000):
    w = rng.uniform(low=-1e2, high=1e2, size=3)
    discount_factor = 1
    
    update_count = 0
    episode_count = 0
    while True:
        epi = Episode(policy, episode_length)
        episode_count += 1
        old_w = w.copy()
        for t in range(epi.length):
            learning_rate = 1e-4 # 1/i # alpha
            measurement = epi.rewardAt(t+1) + discount_factor*v_hat(epi.stateAt(t+1), w) # U_t
            estimate = v_hat(epi.stateAt(t), w)
            grad = getFeatureVector(epi.stateAt(t)) # gradient of (W^T)X is X
            update = learning_rate*(measurement - estimate)*grad
            w = w + update # w_t+1 = w_t + a[U_t-v(s_t, w_t)]*grad(v(s_t, w_t))
            update_count += 1
        
        # Print progress every 100 episode
        # if episode_count%100 == 0: print(f'{update_count=} {w}')
        print(f'{update_count=} {w}')
        if np.linalg.norm(old_w-w) < tolerance: return w

action_space = {
    0: logic.right,
    1: logic.up,
    2: logic.left,
    3: logic.down,
}

# define winning state number as the max state number + 1
# where max state number = 11**16-1
WINNING_STATE = 11**16

class Policy:
    def __init__(self, q) -> None:
        self.q = q
    
    def __call__(self, state: int) -> int:
        possible_actions = get_possible_actions(state)
        q_per_action = np.vectorize(self.q)(state, possible_actions)
        best_action = possible_actions[np.argmax(q_per_action)]
        return best_action
    
class actionValueFunction:
    def __init__(self, w) -> None:
        self.weight = w
        
    def __call__(self, state: int, action: int):
        return q_hat(state, action, self.weight)

def v_hat(state: int, weight: np.ndarray):
    if isTerminalState(state):
        return 0
    else:
        x = getFeatureVector(state)
        return weight.dot(x)
    
def q_hat(state: int, action: int, weight: np.ndarray):
    all_next_states = get_all_next_states(state, action)
    immediate_rewards = []
    next_state_values = []
    for next_state in all_next_states:
        immediate_rewards.append(reward(state, action, next_state))
        next_state_values.append(v_hat(next_state, weight))
    immediate_rewards = np.array(immediate_rewards)
    next_state_values = np.array(next_state_values)
    
    return np.mean(immediate_rewards+next_state_values) # E(r+v(s')) when P(s') is uniform
    
def get_possible_actions(state: int):
    grid = stateToGrid(state)
    all_actions = range(4)
    possible_actions = []
    
    for action in all_actions:
        next_grid, done = action_space[action](grid)
        if done:
            possible_actions.append(action)
    
    possible_actions = np.array(possible_actions)
    return possible_actions
    
def get_all_next_states(state: int, action: int):
    next_states = []
    grid = stateToGrid(state)
    grid, done = action_space[action](grid)
    flat_grid = grid.flatten()
    
    if done:
        for position in range(len(flat_grid)):
            if flat_grid[position] == 0:
                next_grid = logic.add_two(grid, position)
                next_state = gridToState(next_grid)
                next_states.append(next_state)
    else:
        next_states.append(state)
        
    return np.array(next_states)

def reward(current_state: int, current_action: int, next_state: int):
    '''
    Calculate reward based on current state, current action, and next state
    '''
    # reward winning
    if next_state == WINNING_STATE:
        return +10
    
    if 0 <= next_state < WINNING_STATE:
        grid = stateToGrid(next_state)
        # punish losing or choosing an action that does nothing
        if logic.game_state(grid) == 'lose' or current_state == next_state:
            return -10
        # punish valid moves by -1
        else:
            return -1
        
def isTerminalState(state: int):
    if state == WINNING_STATE:
        return True
    
    if 0 <= state < WINNING_STATE:
        grid = stateToGrid(state)
        if logic.game_state(grid) == 'lose':
            return True
        else:
            return False

class Episode:
    def __init__(self, policy, max_length=1000):
        self.state_history = []
        self.action_history = []
        self.reward_history = [None]
        
        initial_grid = logic.new_game(4)
        s = gridToState(initial_grid)
        self.state_history.append(s)
        
        t = 0
        while not isTerminalState(s) and t < max_length:
            a = policy(s)
            # print(a)
            self.action_history.append(a)
            s_prime = transition(s, a)
            r = reward(s, a, s_prime)
            self.reward_history.append(r)
            t += 1
            s = s_prime
            self.state_history.append(s)
        
        self.length = len(self.action_history)
            
    def stateAt(self, t):
        if 0 <= t < len(self.state_history):
            return self.state_history[t]
        else:
            return None
    
    def actionAt(self, t):
        if 0 <= t < len(self.action_history):
            return self.action_history[t]
        else:
            return None
    
    def rewardAt(self, t):
        if 1 <= t < len(self.reward_history):
            return self.reward_history[t]
        else:
            return None

def transition(state: int, action: int):
    grid = stateToGrid(state)
    grid, done = action_space[action](grid)
    
    if done:
        grid = logic.add_two(grid)
        
    return gridToState(grid)

def play():
    grid = logic.new_game(4) # create new game
    while True:
        # show grid
        # each row is printed on a new line because otherwise, 
        # the nested list is printed as a single line
        print(grid)
        # ask user for input
        direction = input(f'Direction (0 right, 1 up, 2 left, 3 down): ')
        if direction not in action_space.keys():
            print(f'Invalid direction. Please input a valid direction.')
            continue
        # transition the grid to next state
        # done flag is used to check if the direction is a valid move
        grid, done = action_space[direction](grid)
        if done:
            # add new tile of value 2 to random position on the grid
            grid = logic.add_two(grid)
            if logic.game_state(grid) == 'win':
                print('Win')
                break
            if logic.game_state(grid) == 'lose':
                print('Lose')
                break

def stateToGrid(state: int, winning_value=2048):
    '''
    converts state number s denoted by an integer 
    in the range [0, 11^16-1] to a 4x4 grid
    '''
    base = int(math.log2(winning_value))
    if not 0 <= state < base**16: return None
    
    # helper function
    def numberToBase(n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]
    
    # Convert state number to 16 digit base 11 representation
    # where each digit represents a tile
    digits = numberToBase(state, base)
    # Pad 0s in the front to get total 16 digits
    zeros = [0] * (16-len(digits))
    grid = zeros + digits
    # Rearrange digits in a 4x4 grid
    grid = np.array(grid).reshape((4,4))
    # Convert digit values to tile values
    grid = np.power(2, grid)
    # Replace all tiles of value 1 with 0 (aka blank tile)
    grid[grid==1] = 0
    
    return grid

def gridToState(grid, winning_value=2048):
    '''
    Hashes a 4x4 grid to state number s in the range [0, 11^16-1]
    This function should be the inverse of stateToGrid()
    
    Works by first converting the grid to a base 11 representation
    then calculating the actual value of the base 11 representation
    '''
    base = int(math.log2(winning_value))
    arr = np.array(grid)
    tile_values = arr.flatten()
    if winning_value in tile_values:
        return WINNING_STATE
    # Replace blank tiles with 1
    arr[arr==0] = 1
    # Convert to base 11 representation
    digits = np.log2(arr.flatten()).astype('int64')
    values_per_digit = np.power(base, np.arange(len(digits)-1, -1, -1, dtype='int64'))
    state = digits.dot(values_per_digit)
    
    return state

def getFeatureVector(state: int):
    '''
    Converts state number to feature vector
    '''
    grid = stateToGrid(state)
    
    return np.array([mean(grid),
                     std(grid),
                     fullness(grid)])

def mean(grid: np.ndarray):
    '''
    Calculates the mean of the tiles on the grid
    '''
    return grid.mean()

def std(grid: np.ndarray):
    '''
    Calculates standard deviation of the tiles on the grid
    '''
    return grid.std()

def fullness(grid: np.ndarray):
    '''
    Calculates how full the grid is
    '''
    return np.count_nonzero(grid)

if __name__ == '__main__':
    main()