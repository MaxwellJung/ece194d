import numpy as np
import random
from reinforcement_learning import Environment, Action, State

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.float_)
class Tetris(Environment):
    piece1 = np.array([[0, 1],
                       [1, 1]])
    piece2 = np.array([[0, 1, 1],
                       [1, 1, 0]])
    piece3 = np.array([[1],
                       [1]])
    
    all_pieces = [piece1, piece2, piece3]
    max_piece_height = max([max(p.shape) for p in all_pieces])
    
    def __init__(self, width=3, height=3, starting_piece=None):
        self.playable_area = (height, width)
        self.line_height = self.playable_area[0]
        self.board = np.zeros(shape=(height+self.max_piece_height,width))
        self.select_next_piece(i=starting_piece, zero_indexed=True)
        
        self.all_boards = [np.pad(bin_array(i, width*height).reshape(self.playable_area), ((self.max_piece_height,0),(0,0))) for i in range(2**(width*height))]
        self.all_states = [State(board=b, piece=p) for b in self.all_boards for p in self.all_pieces]
        self.all_actions = [Action(orientation=o, location=l) for o in range(4) for l in range(width)]
        
        self.score = 0
        self.reward = 0
        self.game_over = False
    
    def set_state(self, s: State):
        self.game_over = False
        self.board = s.kwargs['board']
        self.current_piece = s.kwargs['piece']
    
    def get_state(self):
        if self.game_over:
            return State(terminal=True)
        return State(board=self.board, piece=self.current_piece)
    
    def get_reward(self):
        # if self.game_over:
        #     return 0
        return self.reward
        
    def transition(self, a: Action):
        points = self.place_piece(**a.kwargs)
        self.reward = points
        self.select_next_piece()
        
    def select_next_piece(self, i=None, zero_indexed=True):
        if i is None:
            self.current_piece = random.choice(self.all_pieces)
            return
        if not zero_indexed:
            self.current_piece = self.all_pieces[i-1]
            return
        self.current_piece = self.all_pieces[i]
    
    def place_piece(self, orientation=0, location=0) -> float:
        '''Place current piece using orientation and location.
        Return points acquired from placing the piece'''
        piece = np.rot90(self.current_piece, orientation)
        piece_height = piece.shape[0]
        piece_width = piece.shape[1]
        board_height = self.board.shape[0]
        board_width = self.board.shape[1]
        
        def drop_piece():
            '''Try to drop piece. Raises error if illegal move'''
            pad_top = self.max_piece_height - piece_height
            pad_bottom = self.line_height
            pad_left = location
            pad_right = board_width-pad_left-piece_width
            
            try: piece_mask = np.pad(piece, ((pad_top,pad_bottom),(pad_left,pad_right)))
            except ValueError: raise self.illegalActionException # illegal orientation/location
            
            for h in range(self.line_height, -1, -1): # loop until piece completely drops
                board_buffer = piece_mask + self.board
                collision = board_buffer > 1
                if np.any(collision): break
                lowest_board = board_buffer
                piece_mask = np.roll(piece_mask, shift=1, axis=0)
                
            self.board = lowest_board
        
        def clear_complete_rows():
            complete_rows = np.all(self.board, axis=1)
            points = np.sum(complete_rows)
            incomplete_rows = np.invert(complete_rows)
            cleaned_board = self.board[incomplete_rows]
            self.board = np.pad(cleaned_board, ((board_height-cleaned_board.shape[0],0),(0,0)))
            return points
        
        drop_piece()
        points = clear_complete_rows()
        
        above_line = self.board[:-self.line_height]
        if np.any(above_line): self.game_over = True
        return points
        
    def start_interactive_play(self):
        while not self.game_over:
            print(f'Score: {self.score}')
            print(self.board)
            print(self.current_piece)
            try:
                points = self.place_piece(orientation=int(input('orientation:')), location=int(input('location:')))
            except self.illegalActionException:
                print(f'Invalid orientation/location. Try again.')
            except self.gameOverException:
                self.game_over = True
                print(self.board)
                print(f'Game Over!')
                print(f'Final Score: {self.score}')
            else:
                self.score += points
                self.current_piece = self.select_next_piece()
                
    def __repr__(self) -> str:
        return f'{self.board}\nscore={self.score}'