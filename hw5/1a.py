import numpy as np

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

def main():
    print(transition(start_state, 1))


if __name__ == '__main__':
    main()