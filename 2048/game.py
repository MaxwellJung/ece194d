import numpy as np
from environment import Environment

class TwntyFrtyEight(Environment):
    WINNING_VALUE = 2048
    # define winning state number as the max state number + 1
    WINNING_STATE = np.log2(WINNING_VALUE)**16
    # map state number to board configuration
    visited_states = {}
    
    @staticmethod
    def play():
        board = TwntyFrtyEight.new_board(row_count=4, col_count=4) # create new game
        while True:
            print(board)
            # ask user for input
            try: direction = int(input(f'Direction (0 left, 1 up, 2 right, 3 down): '))
            except ValueError: continue
            if direction not in TwntyFrtyEight.get_valid_actions(board):
                print(f'Invalid direction. Please input a valid direction.')
                continue
            # transition the board to next state
            # done flag is used to check if the direction is a valid move
            board = TwntyFrtyEight.move(board, direction)
            # add new tile of value 2 to random position on the board
            board = TwntyFrtyEight.add_two(board)
            if TwntyFrtyEight.get_status(board) == 'win':
                print('Win')
                break
            if TwntyFrtyEight.get_status(board) == 'lose':
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
    def get_status(board: np.ndarray):
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
    def get_valid_actions(state: int):
        board = TwntyFrtyEight.state_to_board(state)
        all_actions = range(4)
        valid_actions = []
        
        for action in all_actions:
            next_board = TwntyFrtyEight.move(board, action)
            if np.any(board != next_board):
                valid_actions.append(action)
        
        return valid_actions
    
    @staticmethod
    def compress(board: np.ndarray):
        '''
        compress the board towards the left
        '''
        for row_index in range(board.shape[0]):
            row = board[row_index]
            compressed_row = row[row!=0]
            new_row = np.pad(compressed_row, (0, board.shape[1]-len(compressed_row)))
            board[row_index] = new_row
            
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
        
        return next_board
    
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
        base = int(np.log2(TwntyFrtyEight.WINNING_VALUE))
        if np.any(arr==TwntyFrtyEight.WINNING_VALUE): return TwntyFrtyEight.WINNING_STATE
        # Replace blank tiles with 1
        arr[arr==0] = 1
        # Convert to base 11 representation
        digits = np.log2(arr.flatten()).astype('int64')
        values_per_digit = np.power(base, np.arange(len(digits)-1, -1, -1, dtype='int64'))
        state = digits.dot(values_per_digit)
        
        TwntyFrtyEight.visited_states[state] = board
        
        return state
    
    @staticmethod
    def get_feature_vector(state: int):
        '''
        Converts state number to feature vector
        '''
        board = TwntyFrtyEight.state_to_board(state)
        
        def mean(): return board.mean()
        def std(): return board.std()
        def fullness(): return np.count_nonzero(board)

        def distance_to_corner():
            '''
            Calculates manhattan distance of the largest tile to the nearest corner
            '''
            row_count, col_count = board.shape
            corners = np.array([(0,0), (0, col_count-1), (row_count-1, 0), (row_count-1, col_count-1)])
            max_pos = np.unravel_index(np.argmax(board), board.shape)
            return np.min(np.linalg.norm(corners-max_pos, ord=1, axis=1))

        def center_sum(): return np.sum(board[1:-1, 1:-1])
        def perimeter_sum(): return np.sum(board)-center_sum()

        return np.array([1,
                        mean(),
                        std(),
                        fullness(),
                        distance_to_corner(),
                        center_sum(),
                        perimeter_sum()])
        
    @staticmethod
    def get_all_next_states(state: int, action: int):
        next_states = []
        board = TwntyFrtyEight.state_to_board(state)
        board = TwntyFrtyEight.move(board, action)
            
        for position in range(np.prod(board.shape)):
            position = np.unravel_index(position, board.shape)
            if board[position] == 0:
                next_board = TwntyFrtyEight.add_two(board, position)
                next_state = TwntyFrtyEight.board_to_state(next_board)
                next_states.append(next_state)
        else:
            next_states.append(state)
            
        return np.array(next_states)
    
    @staticmethod
    def reward(current_state: int, current_action: int, next_state: int):
        '''
        Calculate reward based on current state, current action, and next state
        '''
        # reward winning
        if next_state == TwntyFrtyEight.WINNING_STATE:
            return +1e4
        
        if 0 <= next_state < TwntyFrtyEight.WINNING_STATE:
            current_board = TwntyFrtyEight.state_to_board(current_state)
            next_board = TwntyFrtyEight.state_to_board(next_state)
            # punish losing or choosing an action that does nothing
            if TwntyFrtyEight.get_status(next_board) == 'lose' or current_state == next_state:
                return -1e4
            # punish valid moves by -1
            else:
                # reward combining tiles
                return np.count_nonzero(current_board)-np.count_nonzero(next_board)
            
    @staticmethod
    def transition(state: int, action: int):
        board = TwntyFrtyEight.state_to_board(state)
        next_board = TwntyFrtyEight.move(board, action)
        
        if np.any(next_board != board):
            board = TwntyFrtyEight.add_two(next_board)
            
        return TwntyFrtyEight.board_to_state(board)
    
    @staticmethod
    def is_terminal_state(state: int):
        if state == TwntyFrtyEight.WINNING_STATE:
            return True
        
        if 0 <= state < TwntyFrtyEight.WINNING_STATE:
            board = TwntyFrtyEight.state_to_board(state)
            if TwntyFrtyEight.get_status(board) == 'lose':
                return True
            else:
                return False