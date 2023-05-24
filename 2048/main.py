import numpy as np
import logic
import math
import logging

# numpy config
rng = np.random.default_rng()
np.set_printoptions(precision=4)

# logging config
logging.basicConfig(level=logging.INFO, 
                    format="%(message)s",
                    handlers=[
                        logging.FileHandler("info.log", mode='w'),
                        logging.StreamHandler()],
                    )

def main():
    w_star = policy_iteration(tolerance=1e-2)
    logging.info(w_star)

def policy_iteration(tolerance):
    w = rng.uniform(low=-1e2, high=1e2, size=len(getFeatureVector(gridToState(logic.new_game(4)))))
    while True:
        old_w = w.copy()
        pi = Policy(weight=w)
        w = sgd(policy=pi, tolerance=1e-3)
        if np.linalg.norm(old_w-w) < tolerance:
            return w
    
def sgd(policy, tolerance, episode_length=2048):
    def print_status():
        logging.info(f'{episode_count=} {update_count=} \n{w}')
        logging.info(f'{stats} win_rate={stats["win"]/episode_count:.2%} average_steps={update_count/episode_count:.2f}')
        start_state = rng.integers(WINNING_STATE)
        logging.info(stateToGrid(start_state))
        logging.info(action_space[policy(start_state)])
        logging.info(v_hat(start_state, w))
    
    w = rng.uniform(low=-1e2, high=1e2, size=len(policy.weight))
    discount = 0.9
    lamb = 0.5
    
    update_count = 0
    episode_count = 0
    stats = {'win': 0,
             'loss': 0,
             'incomplete': 0,}
    
    while True:
        epi = Episode(policy, episode_length)
        episode_count += 1
        old_w = w.copy()
        z = 0
        for t in range(epi.length):
            learning_rate = 1e-7 # alpha
            measurement = epi.rewardAt(t+1) + discount*v_hat(epi.stateAt(t+1), w) # U_t
            estimate = v_hat(epi.stateAt(t), w)
            grad = getFeatureVector(epi.stateAt(t)) # gradient of (W^T)X is X
            z = lamb*discount*z + grad
            update = learning_rate*(measurement - estimate)*z
            w = w + update # w_t+1 = w_t + a[U_t-v(s_t, w_t)]*grad(v(s_t, w_t))
            update_count += 1
            
        # Record stats:
        stats[epi.result] += 1
        
        # logging.info(f'{episode_count=} {update_count=} \n{w}')
        # Print progress every 250 episodes
        if episode_count%250 == 0:
            print_status()
        if np.linalg.norm(old_w-w) < tolerance:
            logging.info(f'------------------------Final convergence------------------------')
            print_status()
            logging.info(f'-----------------------------------------------------------------')
            break
    return w
        
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
    def __init__(self, weight) -> None:
        self.weight = weight
    
    def __call__(self, state: int) -> int:
        possible_actions = get_possible_actions(state)
        q_per_action = [q_hat(action=action, state=state, weight=self.weight) \
                        for action in possible_actions]
        best_action = possible_actions[np.argmax(q_per_action)]
        # logging.info(possible_actions)
        # logging.info(q_per_action)
        # logging.info(best_action)
        return best_action
        # return rng.integers(4)

def v_hat(state: int, weight: np.ndarray):
    if isTerminalState(state):
        return 0
    else:
        x = getFeatureVector(state)
        return weight.dot(x)
    
def q_hat(action: int, state: int, weight: np.ndarray):
    all_next_states = get_all_next_states(state, action)
    rewards = [reward(next_state=next_state, current_state=state, current_action=action) \
             + v_hat(state=next_state, weight=weight) \
               for next_state in all_next_states]
    # logging.info(rewards)
    return np.mean(rewards) # E(r+v(s')) when P(s') is uniform
    
def get_possible_actions(state: int):
    grid = stateToGrid(state)
    all_actions = range(4)
    possible_actions = []
    
    for action in all_actions:
        next_grid, done = action_space[action](grid)
        if done:
            possible_actions.append(action)
            
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

def reward(next_state: int, current_state: int, current_action: int):
    '''
    Calculate reward based on current state, current action, and next state
    '''
    # reward winning
    if next_state == WINNING_STATE:
        return +1e3
    
    if 0 <= next_state < WINNING_STATE:
        grid = stateToGrid(next_state)
        # punish losing or choosing an action that does nothing
        if logic.game_state(grid) == 'lose' or current_state == next_state:
            return -1e3
        # punish valid moves by -1
        else:
            return 0
        
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
    def __init__(self, policy, max_length):
        self.state_history = []
        self.action_history = []
        self.reward_history = [None]
        
        while True:
            s = rng.integers(WINNING_STATE)
            if not isTerminalState(s):
                break
        self.state_history.append(s)
        
        t = 0
        while not isTerminalState(s) and t < max_length:
            a = policy(s)
            self.action_history.append(a)
            s_prime = transition(s, a)
            r = reward(s_prime, s, a)
            self.reward_history.append(r)
            t += 1
            s = s_prime
            self.state_history.append(s)
        
        self.length = len(self.action_history)
        
        final_state = self.state_history[-1]
        if final_state == WINNING_STATE:
            self.result = 'win'
        else:
            if logic.game_state(stateToGrid(final_state)) == 'lose':
                self.result = 'loss'
            else:
                self.result = 'incomplete'
            
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
        logging.info(grid)
        # ask user for input
        direction = input(f'Direction (0 right, 1 up, 2 left, 3 down): ')
        if direction not in action_space.keys():
            logging.info(f'Invalid direction. Please input a valid direction.')
            continue
        # transition the grid to next state
        # done flag is used to check if the direction is a valid move
        grid, done = action_space[direction](grid)
        if done:
            # add new tile of value 2 to random position on the grid
            grid = logic.add_two(grid)
            if logic.game_state(grid) == 'win':
                logging.info('Win')
                break
            if logic.game_state(grid) == 'lose':
                logging.info('Lose')
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
    
    X = np.array([1,
                  mean(grid),
                  std(grid),
                  fullness(grid),
                  distance_to_corner(grid),
                  center_sum(grid),
                  corner_sum(grid),
                  edge_sum(grid),
                  neighbor_difference(grid),])
    
    return X

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

def distance_to_corner(grid: np.ndarray):
    '''
    Calculates manhattan distance of the largest tile to the nearest corner
    '''
    row_count, col_count = grid.shape
    corners = np.array([(0,0), (0, col_count-1), (row_count-1, 0), (row_count-1, col_count-1)])
    max_pos = np.unravel_index(np.argmax(grid), grid.shape)
    return np.min(np.linalg.norm(corners-max_pos, ord=1, axis=1))

def center_sum(grid: np.ndarray):
    '''
    Sum of center tiles (center = tiles excluding the edges)
    '''
    return np.sum(grid[1:-1, 1:-1])

def corner_sum(grid: np.ndarray):
    '''
    Sum of corner tiles
    '''
    row_count = grid.shape[0]
    col_count = grid.shape[1]
    return np.sum([grid[0, 0], grid[row_count-1, 0], grid[0, col_count-1], grid[row_count-1, col_count-1]])

def edge_sum(grid: np.ndarray):
    '''
    Sum of edge tiles
    '''
    return np.sum(grid)-center_sum(grid)-corner_sum(grid)

def neighbor_difference(grid: np.ndarray):
    '''
    Sum of absolute value of differences between neighboring tiles
    '''
    grid[grid==0] = 1
    grid = np.log2(grid)
    row_diff = grid[:-1] - grid[1:]
    col_diff = grid[:, :-1] - grid[:, 1:]
    return np.sum(np.square(row_diff)) + np.sum(np.square(col_diff))



if __name__ == '__main__':
    main()