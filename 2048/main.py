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
    grid = logic.new_game(4)
    while True:
        for row in grid:
            print(row)
        direction = input(f'Direction (left, right, up, down): ')
        if direction not in transition.keys():
            print(f'Invalid direction. Please input a valid direction.')
            continue
        grid, done = transition[direction](grid)
        if done:
            grid = logic.add_two(grid)
            if logic.game_state(grid) == 'win':
                print('Win')
                break
            if logic.game_state(grid) == 'lose':
                print('Lose')
                break
        
        
if __name__ == '__main__':
    main()