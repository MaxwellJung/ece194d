import numpy as np

class gridWorld:
    def __init__(self) -> None:
        self.states = range(11)
        self.current_state = 0
        self.values = np.zeros(len(self.states))
        
        p = np.zeros(shape=(len(self.states), len(self.states)))
        grid = [[2, 4, 7, 10],
                [1, None, 6, 9],
                [0, 3, 5, 8]]
        
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
                    
        self.trap = 9
        self.goal = 10
        p[self.trap] = p[self.goal] = np.zeros(len(self.states))
        
        self.p = p
        
    def update_value(self):
        expected_rewards = np.full(len(self.states), -1)
        expected_rewards[self.goal] = 10
        expected_rewards[self.trap] = -10
        
        self.values = self.p.dot(self.values) + expected_rewards

def main():
    g = gridWorld()
    
    while True:
        old = g.values
        g.update_value()
        new = g.values
        print(new)
        
        if np.linalg.norm(old-new) < 0.001:
            break
        
if __name__ == '__main__':
    main()