from gym_chess import ChessEnvV2, num_to_piece_char
import time, os

K_PLY = 2

def clear_terminal():
  """Clears the terminal"""
  # Use clear command for Linux/Unix systems
  if os.name == 'posix':
    os.system('clear')
  # Use cls command for Windows systems
  else:
    os.system('cls')
    
def get_current_player(curPlayerBool):
    return "WHITE" if curPlayerBool==True else "BLACK"

def get_FEN(board, player_to_move, wk_castle, wq_castle, bk_castle, bq_castle, halfmove_clock, fullmove_number):
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
    fen_string += " " + str(halfmove_clock) + " " + fullmove_number
    
    return fen_string

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
            initial_state = env.reset(current_move_by=cur_player)
            
    return childChessNodes

def get_init_reward(curChessNode):
    if curChessNode.player == True:
        return -1e4
    else:
        return 1e4
    
def eval(FEN):
    """Evaluate Function for the leaf nodes of the game tree"""
    return 1
    
def compute_minimax(root):
    start = time.time()
    for child in root.children:
        root.minimax = max(root.minimax, minimax(child))
    #print(f'Computed Minimax values for the tree in {time.time()-start}s')
          
def minimax(curChessNode):
    if len(curChessNode.children) == 0:
        return eval(curChessNode)
    elif curChessNode.player == True:
        for child in curChessNode.children:
          curChessNode.minimax = max(curChessNode.minimax, minimax(child)) 
        return curChessNode.minimax
    else:
        for child in curChessNode.children:
          curChessNode.minimax = min(curChessNode.minimax, minimax(child)) 
        return curChessNode.minimax
    
def play_chess(env, state, curPlayer, clear_screen=True):
    done = False
    while done != True:
        
        curChessNode = ChessNode(state, None, curPlayer, 0)
        compute_minimax(curChessNode)
        
        while len(curChessNode.children) != 0:
            
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
                        
            curChessNode = bestMove
            state, reward, done, info = env.step(bestAction)
            curPlayer = not curPlayer # opponent's move
            
            if clear_screen:
                clear_terminal()

            env.render()
            
            if done:
                break            
    

if __name__ == "__main__":
    env = ChessEnvV2(opponent='none')
    env.render()
    play_chess(env, env.state, True)
    