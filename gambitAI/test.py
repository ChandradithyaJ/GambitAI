from gym_chess import ChessEnvV2

K_PLY = 3

class ChessNode:
    def __init__(self, state, minimax, player, depth):
        self.state = state # env state
        self.minimax = minimax # float value
        self.player = player # boolean, WHITE - TRUE, BLACK - FALSE
        self.depth = depth
        self.children = get_children(self)
        self.reward = get_init_reward(self)
        
    
def get_children(curChessNode):
    childChessNodes = []
        
    if curChessNode.depth < K_PLY:
        env = ChessEnvV2(opponent='none', initial_board=curChessNode.state['board'])
        moves = env.possible_moves
        actions = [env.move_to_action(move) for move in moves]
        for action in actions:
            new_state, reward, done, info = env.step(action)
            childChessNodes.append(ChessNode(new_state, 1, not curChessNode.player, curChessNode.depth + 1))
            initial_state = env.reset()
            
    return childChessNodes

def get_init_reward(curChessNode):
    if curChessNode.player == True:
        return -1e4
    else:
        return 1e4

def compute_minimax(root):
      for child in root.children:
          root.reward = max(root.reward, minimax(child))
          
def minimax(curChessNode):
    if len(curChessNode.children) == 0:
        return 1
    elif curChessNode.player == True:
        for child in curChessNode.children:
          curChessNode.reward = max(curChessNode.reward, minimax(child)) 
        return curChessNode.reward
    else:
        for child in curChessNode.children:
          curChessNode.reward = min(curChessNode.reward, minimax(child)) 
        return curChessNode.reward