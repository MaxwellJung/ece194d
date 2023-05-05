import numpy as np

class Tetris:
    piece1 = np.array([[0, 1],
                       [1, 1]])
    piece2 = np.array([[0, 1, 1],
                       [1, 1, 0]])
    piece3 = np.array([[1],
                       [1]])
    all_pieces = [piece1, piece2, piece3]
    
    @classmethod
    def select_piece(cls, i, zero_indexed=False):
        if zero_indexed:
            return cls.all_pieces[i]
        return cls.all_pieces[i-1]
    
    def __init__(self, starting_piece):
        self.board = np.zeros(shape=(3,3))
        self.current_piece = self.select_piece(starting_piece)
    
    class invalidMoveException(Exception):
        pass
    
    class gameOverException(Exception):
        pass
    
    def place_piece(self, orientation=0, location=0):
        def drop(piece: np.ndarray, subboard: np.ndarray):
            if piece.shape[1] != subboard.shape[1]:
                raise self.invalidMoveException
            
            for h in range(self.board.shape[0], -1, -1):
                drop_one_block = (np.pad(piece, ((self.board.shape[0]-h,h),(0,0)))) + (np.pad(subboard, ((piece.shape[0],0),(0,0))))
                if np.any((drop_one_block>1)):
                    break
                valid = drop_one_block
                
            if np.any(valid[:-self.board.shape[1]]):
                raise self.gameOverException
            
            return valid[-self.board.shape[1]:, :]
        
        rotated_piece = np.rot90(self.current_piece, orientation)
        piece_width = rotated_piece.shape[1]
        d = drop(rotated_piece, self.board[:, location:location+piece_width])
        self.board[:, location:location+piece_width] = d
        
        print(self.board)