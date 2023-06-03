import numpy as np
from environment import Environment
import features

class TwntyFrtyEight(Environment):
    WINNING_VALUE = 2048
    # define winning state number as the max state number + 1
    WINNING_STATE = np.log2(WINNING_VALUE)**16
    # map state number to board configuration
    visited_states = {}
    
    @staticmethod
    def play():
        board = TwntyFrtyEight.new_board(row_count=4, col_count=4) # create new game
        score = 0
        while True:
            print(board)
            print(f'{score=}')
            # ask user for input
            try: direction = int(input(f'Direction (0 left, 1 up, 2 right, 3 down): '))
            except ValueError: continue
            if direction not in TwntyFrtyEight.get_valid_actions(TwntyFrtyEight.board_to_state(board)):
                print(f'Invalid direction. Please input a valid direction.')
                continue
            # transition the board to next state
            # done flag is used to check if the direction is a valid move
            board, points = TwntyFrtyEight.move(board, direction)
            score += points
            # add new tile of value 2 to random position on the board
            board = TwntyFrtyEight.add_two(board)
            if TwntyFrtyEight.get_board_status(board) == 'win':
                print('Win')
                break
            if TwntyFrtyEight.get_board_status(board) == 'lose':
                print('Lose')
                break
        
    @staticmethod
    def new_board(row_count, col_count):
        empty_board = np.zeros((row_count, col_count), dtype=int)
        board = TwntyFrtyEight.add_two(TwntyFrtyEight.add_two(empty_board))
        return board
        
    @staticmethod
    def add_two(board: np.ndarray, position: tuple[int, int]=None):
        if position is None:
            while True:
                position = np.unravel_index(np.random.randint(np.prod(board.shape)), board.shape)
                if board[position] == 0:
                    break
        next_board = np.copy(board)
        next_board[position] = 2
        return next_board
    
    @staticmethod
    def get_board_status(board: np.ndarray):
        # check for win cell
        if np.any(board==2048):
            return 'win'
        # check for any zero entries
        if np.any(board==0):
            return 'not over'
        # check for same cells that touch each other
        for i in range(len(board)-1):
            # intentionally reduced to check the row on the right and below
            # more elegant to use exceptions but most likely this will be their solution
            for j in range(len(board[0])-1):
                if board[i][j] == board[i+1][j] or board[i][j+1] == board[i][j]:
                    return 'not over'
        for k in range(len(board)-1):  # to check the left/right entries on the last row
            if board[len(board)-1][k] == board[len(board)-1][k+1]:
                return 'not over'
        for j in range(len(board)-1):  # check up/down entries on last column
            if board[j][len(board)-1] == board[j+1][len(board)-1]:
                return 'not over'
        return 'lose'
    
    @staticmethod
    def get_state_status(state: int):
        board = TwntyFrtyEight.state_to_board(state)
        return np.max(board)
    
    @staticmethod
    def get_valid_actions(state: int):
        board = TwntyFrtyEight.state_to_board(state)
        all_actions = range(4)
        valid_actions = []
        
        for action in all_actions:
            next_board, points = TwntyFrtyEight.move(board, action)
            if np.any(board != next_board):
                valid_actions.append(action)
        
        return valid_actions
    
    @staticmethod
    def compress(board: np.ndarray):
        '''
        compress the board towards the left
        vectorized algorithm from
        https://stackoverflow.com/a/43011036
        '''
        valid_mask = board!=0
        flipped_mask = valid_mask.sum(1,keepdims=1) > np.arange(board.shape[1]-1,-1,-1)
        flipped_mask = flipped_mask[:,::-1]
        board[flipped_mask] = board[valid_mask]
        board[~flipped_mask] = 0
            
        return board
    
    @staticmethod
    def merge(board: np.ndarray):
        points = 0
        for i in range(board.shape[0]):
            for j in range(board.shape[1]-1):
                if board[i][j] != 0 and board[i][j] == board[i][j+1]:
                    board[i][j] *= 2
                    points += board[i][j]
                    board[i][j+1] = 0
        return board, points
    
    @staticmethod
    def move(board: np.ndarray, action: int):
        '''
        helper function for right, up, left, down functions below
        '''
        next_board = np.copy(board)
        # rotate board
        next_board = np.rot90(next_board, action)
        # compress, merge, compress towards left
        next_board = TwntyFrtyEight.compress(next_board)
        next_board, points = TwntyFrtyEight.merge(next_board)
        next_board = TwntyFrtyEight.compress(next_board)
        # rotate back to original
        next_board = np.rot90(next_board, -action)
        
        return next_board, points
    
    @staticmethod
    def left(board: np.ndarray):
        '''
        Shift Board Left
        '''
        return TwntyFrtyEight.move(board, action=0)
    
    @staticmethod
    def up(board: np.ndarray):
        '''
        Shift Board Up
        '''
        return TwntyFrtyEight.move(board, action=1)
    
    @staticmethod
    def right(board: np.ndarray):
        '''
        Shift Board Up
        '''
        return TwntyFrtyEight.move(board, action=2)
    
    @staticmethod
    def down(board: np.ndarray):
        '''
        Shift Board Down
        '''
        return TwntyFrtyEight.move(board, action=3)
    
    @staticmethod
    def get_initial_state():
        return TwntyFrtyEight.board_to_state(TwntyFrtyEight.new_board(4, 4))
    
    @staticmethod
    def state_to_board(state: int):
        '''
        converts state number s denoted by an integer 
        in the range [0, 11^16-1] to a 4x4 board
        '''
        if state in TwntyFrtyEight.visited_states:
            return TwntyFrtyEight.visited_states[state]
        
        base = int(np.log2(TwntyFrtyEight.WINNING_VALUE))
        if not 0 <= state < TwntyFrtyEight.WINNING_STATE: return None
        
        # Convert state number to 16 digit base 11 representation
        # where each digit represents a tile
        digits = [int(digit, base) for digit in np.base_repr(state, base).zfill(16)]
        board = digits
        # Rearrange digits in a 4x4 grid
        board = np.array(board).reshape((4,4))
        # Convert digit values to tile values
        board = np.power(2, board)
        # Place blank tiles
        board[board==1] = 0
        TwntyFrtyEight.visited_states[state] = board
        
        return board
    
    @staticmethod
    def board_to_state(board):
        '''
        Hashes a 4x4 board to state number s in the range [0, 11^16-1]
        This function should be the inverse of stateToGrid()
        
        Works by first converting the grid to a base 11 representation
        then calculating the actual value of the base 11 representation
        '''
        arr = np.copy(board)
        if np.any(board==TwntyFrtyEight.WINNING_VALUE): return TwntyFrtyEight.WINNING_STATE
        # Replace blank tiles with 1
        arr[arr==0] = 1
        # Convert to base 11 representation
        digits = np.log2(arr.flatten()).astype('int64')
        base = int(np.log2(TwntyFrtyEight.WINNING_VALUE))
        values_per_digit = np.power(base, np.arange(len(digits)-1, -1, -1, dtype='int64'))
        state = digits.dot(values_per_digit)
        
        TwntyFrtyEight.visited_states[state] = board
        
        return state
    
    @staticmethod
    def get_feature_vector(state: int, action: int):
        '''
        Converts state number to feature vector
        '''
        board = TwntyFrtyEight.state_to_board(state)
        compressed_board, points = TwntyFrtyEight.move(board, action)

        S = np.array([
                      points,
                      features.empty_tiles(compressed_board),
                      features.roughness(compressed_board),
                      features.monotonicity(compressed_board),
                      features.mean_vertical_dif(compressed_board),
                      features.mean_horizontal_dif(compressed_board),
                      features.std_vertical_dif(compressed_board),
                      features.std_horizontal_dif(compressed_board),
                      features.snake1(compressed_board),
                      features.snake2(compressed_board),
                      features.snake3(compressed_board),
                      features.snake4(compressed_board),
                    ])
        
        def compute_fourier_basis(features: np.ndarray, n=2):
            '''
            Compute fourier basis of length (n+1)^k where k is the number of featuresnb
            and n is the resolution of fourier series
            '''
            k = len(features)
            fourier_basis = np.zeros((n+1)**k)
            a = np.zeros((k, n+1)) + np.arange(n+1)
            c = np.array(np.meshgrid(*a)).T.reshape(-1,k) #3 features
            fourier_basis = np.cos(np.pi*c.dot(features))
            return fourier_basis
        
        # X = compute_fourier_basis(S, n=5)
        X = S 
        
        return X
    
    @staticmethod
    def reward(current_state: int, current_action: int, next_state: int):
        '''
        Calculate reward based on current state, current action, and next state
        '''
        # reward winning
        if next_state == TwntyFrtyEight.WINNING_STATE:
            return 1000
        
        if 0 <= next_state < TwntyFrtyEight.WINNING_STATE:
            current_board = TwntyFrtyEight.state_to_board(current_state)
            current_max = np.max(current_board)
            next_board = TwntyFrtyEight.state_to_board(next_state)
            next_max = np.max(next_board)
            # punish losing or choosing an action that does nothing
            if TwntyFrtyEight.get_board_status(next_board) == 'lose' or current_state == next_state:
                return -1000
            else:
                # reward combining tiles
                compressed_board, points = TwntyFrtyEight.move(current_board, current_action)
                bonus = next_max*(np.count_nonzero(next_board==0) - np.count_nonzero(current_board==0))
                r = points + bonus
                return r
            
    @staticmethod
    def transition(state: int, action: int):
        board = TwntyFrtyEight.state_to_board(state)
        next_board, points = TwntyFrtyEight.move(board, action)
        
        if np.any(next_board != board):
            board = TwntyFrtyEight.add_two(next_board)
            
        return TwntyFrtyEight.board_to_state(board)
    
    @staticmethod
    def is_terminal_state(state: int):
        if state == TwntyFrtyEight.WINNING_STATE:
            return True
        
        if 0 <= state < TwntyFrtyEight.WINNING_STATE:
            board = TwntyFrtyEight.state_to_board(state)
            if TwntyFrtyEight.get_board_status(board) == 'lose':
                return True
            else:
                return False