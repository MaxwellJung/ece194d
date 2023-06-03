import numpy as np

def mean(board: np.ndarray):
    board = np.copy(board)
    board[board==0] = 1
    board = np.log2(board)
    return board.mean()

def std(board: np.ndarray):
    board = np.copy(board)
    board[board==0] = 1
    board = np.log2(board)
    return board.std()

def empty_tiles(board: np.ndarray): return np.count_nonzero(board==0)

def roughness(board: np.ndarray):
    '''
    Calculates how rough/bumpy the board is
    smoothness formula from https://stats.stackexchange.com/q/24607
    '''
    board = np.copy(board)
    board[board==0] = 1
    board = np.log2(board)
    deltas = np.concatenate([np.diff(board, axis=0).flatten(), np.diff(board, axis=1).flatten()])
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

def distance_to_corner(board: np.ndarray):
    '''
    Calculates manhattan distance of the largest tile to the nearest corner
    '''
    row_count, col_count = board.shape
    corners = np.array([(0,0), (0, col_count-1), (row_count-1, 0), (row_count-1, col_count-1)])
    max_pos = np.unravel_index(np.argmax(board), board.shape)
    return np.min(np.linalg.norm(corners-max_pos, ord=1, axis=1))

def duplicates(board: np.ndarray):
    board = np.copy(board)
    board[board==0] = 1
    board = np.log2(board)
    unique, count = np.unique(board, return_counts=True)
    duplicate, duplicate_count = unique[count>1], count[count>1]
    duplicate, duplicate_count = duplicate[duplicate>0], duplicate_count[duplicate>0]
    duplicate = 1/duplicate
    return duplicate.dot(duplicate_count)