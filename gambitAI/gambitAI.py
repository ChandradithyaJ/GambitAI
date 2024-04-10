from gym_chess import ChessEnvV2
from gym_chess.envs.chess_v2 import get_num_to_piece_char
from stockfish import Stockfish
from utils import clear_terminal, get_num_pieces_on_board, get_current_player
import time

K_PLY = 2
stockfish = Stockfish("stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")
stockfish.set_elo_rating(2830) # Magnus' Classical FIDE rating as of April 10 2024

def get_FEN(board, player_to_move, wk_castle, wq_castle, bk_castle, bq_castle, halfmove_clock, fullmove_number):
    num_to_piece_char = get_num_to_piece_char()
    
    fen_string = ""
    # encode the piece placement
    for idx, row in enumerate(board):
        empty_squares = 0
        for piece in row:
            if piece == 0:
                empty_squares += 1
            else:
                if empty_squares > 0:
                    fen_string += str(empty_squares)
                    empty_squares = 0
                # uppercase letters for white and lowercase letters for black 
                fen_string += num_to_piece_char[piece] if piece > 0 else num_to_piece_char[abs(piece)].lower()
        if empty_squares > 0:
            fen_string += str(empty_squares)
        fen_string += "/" if idx != 7 else ""
        
    # encode current player
    fen_string += " " + player_to_move[0].lower()
    
    # encode castling rights
    castling_string = ""
    no_castling_available = True
    if wk_castle:
        castling_string += "K"
        no_castling_available = False
    if wq_castle:
        castling_string += "Q"
        no_castling_available = False
    if bk_castle:
        castling_string += "k"
        no_castling_available = False
    if bq_castle:
        castling_string += "q"
        no_castling_available = False
    if no_castling_available:
        castling_string += "-"
    fen_string += " " + castling_string
    
    # encode the enpassant possibility (not considered now)
    fen_string += " " + "-"
    
    # encode the halfmove clock and the full move number
    fen_string += " " + str(halfmove_clock) + " " + str(fullmove_number)
    
    return fen_string

def get_req_data_for_FEN(halfmove_clock, curChessNode, piece_moved, num_pieces_on_board, halfmove_number):
    
    # halfmove clock checks:
    # 1. piece captures
    cur_num_pieces = get_num_pieces_on_board(curChessNode.state['board'])
    new_hfc = halfmove_clock + 1 # update
    if num_pieces_on_board != cur_num_pieces:
        new_hfc = 0 # reset
    # 2. pawn advance
    if abs(piece_moved) == 6:
        new_hfc = 0 # reset
        num_pieces_on_board = cur_num_pieces
    
    return new_hfc, num_pieces_on_board, halfmove_number+1

class ChessNode:
    def __init__(self, state, action, player, depth):
        self.state = state # env state
        self.action = action # action made to get to the state
        self.player = player # boolean, WHITE - TRUE, BLACK - FALSE
        self.depth = depth
        self.children = get_children(self)
        self.minimax = get_init_reward(self)
        
def get_children(curChessNode):
    childChessNodes = []
    cur_player = get_current_player(curChessNode.player)
        
    if curChessNode.depth <= K_PLY:
        env = ChessEnvV2(
            opponent='none',
            initial_board=curChessNode.state['board'],
            current_move_by=cur_player
        )
        moves = env.possible_moves
        actions = [env.move_to_action(move) for move in moves]
        for action in actions:
            new_state, reward, done, info = env.step(action)
            childChessNodes.append(ChessNode(new_state, action, not curChessNode.player, curChessNode.depth + 1))
            _ = env.reset(current_move_by=cur_player)
            
    return childChessNodes

def get_init_reward(curChessNode):
    if curChessNode.player == True:
        return -1e9
    else:
        return 1e9
    
def eval(curChessNode, halfmove_clock, halfmove_number):
    """Evaluation function for the leaf nodes of the game tree"""
    state_FEN = get_FEN(
            curChessNode.state['board'],
            get_current_player(curChessNode.player),
            curChessNode.state['white_king_castle_is_possible'],
            curChessNode.state['white_queen_castle_is_possible'],
            curChessNode.state['black_king_castle_is_possible'],
            curChessNode.state['black_queen_castle_is_possible'],
            halfmove_clock,
            int(halfmove_number/2) + 1
        )
    stockfish.set_fen_position(state_FEN)
    return stockfish.get_evaluation()["value"]
    
def compute_minimax(root, halfmove_clock, num_pieces_on_board, halfmove_number):
    start = time.time()
    for child in root.children:
        hmc = halfmove_clock
        npb = num_pieces_on_board
        hmn = halfmove_number
        
        moved_piece_loc = env.action_to_move(child.action)[0]
        piece_moved = child.state['board'][moved_piece_loc[0]][moved_piece_loc[1]]
        
        hmc, npb, hmn = get_req_data_for_FEN(
            hmc,
            child,
            piece_moved,
            npb,
            hmn
        )
        
        if root.player == True:
            root.minimax = max(
                            root.minimax, 
                            minimax(child, hmc, npb, hmn)
                            )
        else:
            root.minimax = min(
                            root.minimax, 
                            minimax(child, hmc, npb, hmn)
                            )
    #print(f'Computed Minimax values for the tree in {time.time()-start}s')
          
def minimax(curChessNode, halfmove_clock, num_pieces_on_board, halfmove_number):
    if len(curChessNode.children) == 0:
        return eval(curChessNode, halfmove_clock, halfmove_number)
    elif curChessNode.player == True:
        for child in curChessNode.children:
            hmc = halfmove_clock
            npb = num_pieces_on_board
            hmn = halfmove_number
            
            moved_piece_loc = env.action_to_move(child.action)[0]
            piece_moved = child.state['board'][moved_piece_loc[0]][moved_piece_loc[1]]
            
            hmc, npb, hmn = get_req_data_for_FEN(
                hmc,
                child,
                piece_moved,
                npb,
                hmn
            )
            curChessNode.minimax = max(
                                curChessNode.minimax, 
                                minimax(child, hmc, npb, hmn)
                            )
            return curChessNode.minimax
    else:
        for child in curChessNode.children:
            hmc = halfmove_clock
            npb = num_pieces_on_board
            hmn = halfmove_number
            
            moved_piece_loc = env.action_to_move(child.action)[0]
            piece_moved = child.state['board'][moved_piece_loc[0]][moved_piece_loc[1]]
            
            hmc, npb, hmn = get_req_data_for_FEN(
                hmc,
                child,
                piece_moved,
                npb,
                hmn
            )
            curChessNode.minimax = min(
                                curChessNode.minimax, 
                                minimax(child, hmc, npb, hmn)
                            )
            return curChessNode.minimax
    
def play_chess(env, state, curPlayer, clear_screen=True):
    done = False
    halfmove_clock = 0 # number of halfmoves since the last capture or pawn advance
    halfmove_number = 0 # half moves finished
    num_pieces_on_board = 32 # maintaing a count to identify piece captures
    while done != True:
        
        curChessNode = ChessNode(state, None, curPlayer, 0)

        compute_minimax(
            curChessNode,
            halfmove_clock,
            num_pieces_on_board,
            halfmove_number
        )
            
        if clear_screen:
            # time for the user to see the screen before it is cleared
            time.sleep(1)
            
        bestMoveScore = curChessNode.children[0].minimax
        bestMove = curChessNode.children[0]
        bestAction = curChessNode.children[0].action
            
        curChessNodeChildren = curChessNode.children
            
        if curChessNode.player == True:
            for child in curChessNodeChildren:
                if bestMoveScore < child.minimax:
                    bestMoveScore = child.minimax
                    bestMove = child
                    bestAction = child.action
            
        else: 
            for child in curChessNodeChildren:
                if bestMoveScore > child.minimax:
                    bestMoveScore = child.minimax
                    bestMove = child
                    bestAction = child.action
                        
        # infer which piece has been moved
        moved_piece_loc = env.action_to_move(bestAction)[0]
        piece_moved = curChessNode.state['board'][moved_piece_loc[0]][moved_piece_loc[1]]          
            
        curChessNode = bestMove
        state, reward, done, info = env.step(bestAction)
        curPlayer = not curPlayer # opponent's move
            
        if clear_screen:
            clear_terminal()

        env.render()
            
        halfmove_clock, num_pieces_on_board, halfmove_number = get_req_data_for_FEN(
            halfmove_clock,
            curChessNode,
            piece_moved,
            num_pieces_on_board,
            halfmove_number
        )
        
        del curChessNode # clear cache
            
        if done: # game over
            break            
    

if __name__ == "__main__":
    env = ChessEnvV2(opponent='none')
    env.render()
    play_chess(env, env.state, True)
    