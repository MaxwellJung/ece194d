import numpy as np

class gridWorld:
    def __init__(self) -> None:
        # state soace
        self.states = range(11)
        # values for each state
        self.values = np.zeros(len(self.states))
        
        # transition probability matrix
        p = np.zeros(shape=(len(self.states), len(self.states)))
        grid = [[2, 4, 7, 10],
                [1, None, 6, 9],
                [0, 3, 5, 8]]
        
        # based on grid above, calculate the transition probilities for each state
        for row_num, row in enumerate(grid):
            for col_num, current_state in enumerate(row):
                if current_state is None:
                    continue
                left = grid[row_num][col_num-1] if 0 <= col_num-1 <= 3 else None
                right = grid[row_num][col_num+1] if 0 <= col_num+1 <= 3 else None
                up = grid[row_num-1][col_num] if 0 <= row_num-1 <= 2 else None
                down = grid[row_num+1][col_num] if 0 <= row_num+1 <= 2 else None
                
                actions = [right, up, left, down]
                for next_state in actions:
                    if next_state is None:
                        next_state = current_state
                    p[current_state, next_state] += 1/len(actions)
        
        # set transition probabilities of trap and goal to 0
        # meaning there is no transition out of trap/goal
        self.trap = 9
        self.goal = 10
        p[self.trap] = p[self.goal] = np.zeros(len(self.states))
        
        self.transition_p = p
        
    # update values array using Bellman's equation
    def update_value(self):
        # expected reward for picking a random action
        # all states have -1 expected reward, except for goal (+10) and trap (-10)
        expected_rewards = np.full(len(self.states), -1)
        expected_rewards[self.goal] = 10
        expected_rewards[self.trap] = -10
        
        # e.g. v(0) = 0.25(-1 + v(3)) + 0.25(-1 + v(1)) + 0.25(-1 + v(0)) + 0.25(-1 + v(0))
        #           = 0.25*v(3) + 0.25*v(1) + 0.5*v(0) - 1
        self.values = self.transition_p.dot(self.values) + expected_rewards

def main():
    g = gridWorld()
    
    # continuously update value until the distance is less than 0.001 
    while True:
        old = g.values
        g.update_value()
        new = g.values
        print(new)
        
        if np.linalg.norm(old-new) < 0.001:
            break
        
if __name__ == '__main__':
    main()