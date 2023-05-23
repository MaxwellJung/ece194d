import numpy as np
import logic
import math

def main():
    # play()
    for s in range(11**16):
        grid = stateToGrid(s)
        state = gridToState(grid)
        # print(f'{s=} {state=} {s==state=}')
        x = featureExtractor(state=state)
        print(grid)
        print(x)
        if s != state:
            print(s)
            break
    
transition = {
    'left': logic.left,
    'right': logic.right,
    'up': logic.up,
    'down': logic.down
}

def play():
    grid = logic.new_game(4) # create new game
    while True:
        # show grid
        # each row is printed on a new line because otherwise, 
        # the nested list is printed as a single line
        for row in grid:
            print(row)
        # ask user for input
        direction = input(f'Direction (left, right, up, down): ')
        if direction not in transition.keys():
            print(f'Invalid direction. Please input a valid direction.')
            continue
        # transition the grid to next state
        # done flag is used to check if the direction is a valid move
        grid, done = transition[direction](grid)
        if done:
            # add new tile of value 2 to random position on the grid
            grid = logic.add_two(grid)
            if logic.game_state(grid) == 'win':
                print('Win')
                break
            if logic.game_state(grid) == 'lose':
                print('Lose')
                break

def stateToGrid(state: int, terminal_value=2048):
    '''
    converts state number s denoted by an integer 
    in the range [0, 11^16-1] to a 4x4 grid
    '''
    base = int(math.log2(terminal_value))
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
    
    return grid.tolist()

def gridToState(grid, terminal_value=2048):
    '''
    Hashes a 4x4 grid to state number s in the range [0, 11^16-1]
    This function should be the inverse of stateToGrid()
    
    Works by first converting the grid to a base 11 representation
    then calculating the actual value of the base 11 representation
    '''
    base = int(math.log2(terminal_value))
    a = np.array(grid)
    # Replace blank tiles with 1
    a[a==0] = 1
    # Convert to base 11 representation
    digits = np.log2(a.flatten())
    values_per_digit = np.power(base, np.arange(len(digits)-1, -1, -1, dtype='int64'))
    state = digits.dot(values_per_digit)
    
    return state

def featureExtractor(grid=None, state: int=0):
    '''
    converts grid to feature vector
    or alternatively, converts state number to feature vector
    '''
    if grid is None: grid = stateToGrid(state)
    
    return np.array([mean(grid),
                     std(grid),])

def mean(grid):
    '''
    Calculates the mean of the tiles on the grid
    '''
    return np.array(grid).mean()

def std(grid):
    '''
    Calculates standard deviation of the tiles on the grid
    '''
    return np.array(grid).std()

if __name__ == '__main__':
    main()