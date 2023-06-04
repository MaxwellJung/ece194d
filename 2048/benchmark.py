from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from game import TwntyFrtyEight
from agent import Agent

game = TwntyFrtyEight()
agent = Agent(game)
agent.w = np.load('w_star.npy')
features = ['points', 
            'emptiness',
            'roughness',
            'monotonicity',
            'std_vertical_dif',
            'std_horizontal_dif',
            'tile_delta',
            'mean',
            'std',
            'distance_to_corner',
            'center_sum',
            'perimeter_sum',
            'max_tile',]

def main():
    df = pd.DataFrame({'features': features, 
                       'weights': agent.w}).set_index('features')
    df = df.iloc[::-1]
    df['positive'] = df['weights'] > 0
    df['weights'].plot(kind='barh',
                       color=df.positive.map({True: 'b', False: 'r'}))
    # plt.show()
    
    try: df = pd.read_csv('benchmark.csv')
    except FileNotFoundError: df = generate_data(sample_size=10)
    ax = df.groupby(by='highest_tile').count().plot(kind='bar')
    for container in ax.containers:
        ax.bar_label(container)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()
    
def get_results(i):
    board = game.new_board(row_count=4, col_count=4) # create new game
    S = game.board_to_state(board)
    score = 0
    while True:
        direction = agent.greedy_policy(S)
        # transition the board to next state
        # done flag is used to check if the direction is a valid move
        board, points = game.move(board, direction)
        score += points
        # add new tile of value 2 to random position on the board
        board = game.add_two(board)
        S = game.board_to_state(board)
        if game.is_terminal_state(S):
            break
    
    return np.max(board), score

def generate_data(sample_size=1000):    
    data = {'highest_tile': [],
            'score': []}
    
    with Pool() as pool:
        data = pool.map(get_results, range(sample_size))
    
    df = pd.DataFrame(data, columns=['highest_tile', 'score'])
    df.to_csv('benchmark.csv', index=False)
        
    return df
    
if __name__ == '__main__':
    main()