import numpy as np

def mean(board: np.ndarray):
    return board.mean()

def std(board: np.ndarray):
    return board.std()

def center_sum(board: np.ndarray): 
    return np.sum(board[1:-1, 1:-1])

def perimeter_sum(board: np.ndarray): 
    return np.sum(board)-center_sum(board)

def emptiness(board: np.ndarray):
    '''
    1 is fully empty, 0 is full
    '''
    return np.count_nonzero(board==0)

def roughness(board: np.ndarray):
    '''
    Calculates how rough/bumpy the board is
    smoothness formula from https://stats.stackexchange.com/q/24607
    '''
    matrix = np.copy(board)
    matrix[matrix==0] = 1
    matrix = np.log2(matrix)
    deltas = np.concatenate([np.diff(matrix, axis=0).flatten(), np.diff(matrix, axis=1).flatten()])
    roughness = np.std(deltas)
    return roughness

def monotonicity(board: np.ndarray):
    '''
    Find length of longest monotonically increasing chain of tiles
    algorithm from https://leetcode.com/problems/longest-increasing-path-in-a-matrix/solutions/78334/python-solution-memoization-dp-288ms/
    '''
    matrix = np.copy(board)
    matrix[matrix==0] = 1
    matrix = np.log2(matrix)
    
    def dfs(i, j):
        if not dp[i][j]:
            val = matrix[i][j]
            dp[i][j] = 1 + max(
                dfs(i - 1, j) if i and val > matrix[i - 1][j] else 0,
                dfs(i + 1, j) if i < M - 1 and val > matrix[i + 1][j] else 0,
                dfs(i, j - 1) if j and val > matrix[i][j - 1] else 0,
                dfs(i, j + 1) if j < N - 1 and val > matrix[i][j + 1] else 0)
        return dp[i][j]

    # if not matrix or not matrix[0]: return 0
    
    M, N = len(matrix), len(matrix[0])
    dp = [[0] * N for i in range(M)]
    longest_path_length = max(dfs(x, y) for x in range(M) for y in range(N))
    
    return longest_path_length

def max_tile(board: np.ndarray):
    max_tile_value = np.max(board)
    return np.log2(max_tile_value) if max_tile_value > 0 else 0

def tile_delta(board: np.ndarray):
    '''
    Find difference between tiles in horizontal and vertical direction
    Return whichever has lower difference
    '''
    row_dif = np.sum(np.diff(board, axis=0))
    col_dif = np.sum(np.diff(board, axis=1))
    return np.min([row_dif, col_dif])

def distance_to_corner(board: np.ndarray):
    '''
    Calculates manhattan distance of the largest tile to the nearest corner
    '''
    row_count, col_count = board.shape
    corners = np.array([(0,0), (0, col_count-1), (row_count-1, 0), (row_count-1, col_count-1)])
    max_pos = np.unravel_index(np.argmax(board), board.shape)
    return np.min(np.linalg.norm(corners-max_pos, ord=1, axis=1))

def std_vertical_dif(board: np.ndarray):
    return np.std(np.diff(board, axis=0))

def std_horizontal_dif(board: np.ndarray):
    return np.std(np.diff(board, axis=1))

def duplicates(board: np.ndarray):
    unique, count = np.unique(board, return_counts=True)
    duplicate, count = unique[count>1], count[count>1]-1
    return duplicate, count

def count_changes(board1, board2):
    return np.count_nonzero(board1!=board2)

def count_mergeable(board):
    vertical_dif = np.diff(board, axis=0)[(board[:-1]!=0) & (board[1:]!=0)]
    horizontal_dif = np.diff(board, axis=1)[(board[:, :-1]!=0) & (board[:, 1:]!=0)]
    neighbor_dif = np.concatenate([vertical_dif, horizontal_dif])
    return np.count_nonzero(neighbor_dif==0)

def snake_dif(board: np.ndarray):
    '''
    Standard Deviation of differences in snaking sequence of tiles starting from top left to bottom left
    '''
    sequence = np.concatenate([board[0], board[1, ::-1], board[2], board[3, ::-1]])
    return np.std(np.diff(sequence))

def snake1(board: np.ndarray):
    return snake_dif(board)

def snake2(board: np.ndarray):
    return snake_dif(np.rot90(board))

def snake3(board: np.ndarray):
    return snake_dif(np.rot90(board, 2))

def snake4(board: np.ndarray):
    return snake_dif(np.rot90(board, 3))

def snake5(board: np.ndarray):
    return snake_dif(board.T)

def snake6(board: np.ndarray):
    return snake_dif(np.rot90(board.T))

def snake7(board: np.ndarray):
    return snake_dif(np.rot90(board.T, 2))

def snake8(board: np.ndarray):
    return snake_dif(np.rot90(board.T, 3))