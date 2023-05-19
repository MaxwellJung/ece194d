import numpy as np
import logic
import math

def main():
    # play()
    grid = stateToGrid(67206423060)
    print(grid)
    
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
    converts state s denoted by an integer in the range [0, 11^16-1]
    to a 4x4 grid
    '''
    max_exp = int(math.log2(terminal_value))
    
    if not 0 <= state < max_exp**16:
        return None
    
    # helper function
    def numberToBase(n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]
    
    # convert state to a 16-digit base 11 number
    # 16 because there's 16 spaces
    # 11 because there are 11 possible values per space (blank, 2, 4, 8, ... , 1024)
    # reach 2048 on any tile will terminate the game
    digits = numberToBase(state, max_exp)
    zeros = [0] * (16-len(digits))
    grid = zeros + digits
    grid = np.array(grid).reshape((4,4))
    grid = np.power(2, grid)
    grid[grid==1] = 0
    grid = grid.tolist()
    
    return grid

if __name__ == '__main__':
    main()