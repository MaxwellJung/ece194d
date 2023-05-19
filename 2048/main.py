import numpy as np
import logic

def main():
    play()
    
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
        
        
if __name__ == '__main__':
    main()