import numpy as np
import sys

state_count = 6
values = np.zeros(state_count)
denominations = [1, 3]

def update_values():
    pass

def coins(c):
    for i in range(10):
        update_values()
        
    return values[c]

def main(args):
    goal = 63
    print(f'You need {coins(goal)} coins for {goal} cents')

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
