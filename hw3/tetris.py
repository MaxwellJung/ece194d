import numpy as np
import random

class Tetris:
    piece1 = np.array([[0, 1],
                       [1, 1]])
    piece2 = np.array([[0, 1, 1],
                       [1, 1, 0]])
    piece3 = np.array([[1],
                       [1]])
    
    all_pieces = [piece1, piece2, piece3]
    
    def __init__(self, width=3, height=3, starting_piece=None):
        self.board_width = width
        self.board_height = height
        self.board = np.zeros(shape=(height,width))
        self.current_piece = self.select_next_piece(i=starting_piece, zero_indexed=True)
    
    @classmethod
    def select_next_piece(cls, i=None, zero_indexed=True):
        if i is None:
            return random.choice(cls.all_pieces)
        if not zero_indexed:
            return cls.all_pieces[i-1]
        return cls.all_pieces[i]
    
    class invalidMoveException(Exception):
        pass
    
    class gameOverException(Exception):
        pass
    
    def place_piece(self, orientation=0, location=0):
        def drop(piece: np.ndarray):
            piece_height = piece.shape[0]
            piece_width = piece.shape[1]
            pad_left = location
            pad_right = self.board_width-pad_left-piece_width
            try:
                piece_buffer = np.pad(piece, ((0,self.board_height),(pad_left,pad_right)))
            except ValueError:
                raise self.invalidMoveException
            
            for h in range(self.board_height+1):
                drop_one_block = piece_buffer + (np.pad(self.board, ((piece_height,0),(0,0))))
                
                collision = drop_one_block>1
                if np.any(collision):
                    break
                lowest_drop = drop_one_block
                piece_buffer = np.roll(piece_buffer, shift=1, axis=0)
                
            return lowest_drop
        
        def clear_complete_rows(board: np.ndarray):
            complete_rows = np.all(board, axis=1)
            points = np.sum(complete_rows)
            incomplete_rows = np.invert(complete_rows)
            new_board = np.pad(board[incomplete_rows], ((board.shape[0]-board[incomplete_rows].shape[0],0),(0,0)))
            return new_board, points
        
        rotated_piece = np.rot90(self.current_piece, orientation)
        b = drop(rotated_piece)
        b, reward = clear_complete_rows(b)
        
        above_line = b[:-self.board_height]
        if np.any(above_line):
            raise self.gameOverException
        
        self.board = b[-self.board_height:]
