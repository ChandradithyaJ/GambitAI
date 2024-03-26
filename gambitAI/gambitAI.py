from gym_chess import ChessEnvV2

import time

K_PLY = 3

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
        
    if curChessNode.depth < K_PLY:
        env = ChessEnvV2(opponent='none', initial_board=curChessNode.state['board'])
        moves = env.possible_moves
        actions = [env.move_to_action(move) for move in moves]
        for action in actions:
            new_state, reward, done, info = env.step(action)
            childChessNodes.append(ChessNode(new_state, action, not curChessNode.player, curChessNode.depth + 1))
            initial_state = env.reset()
            
    return childChessNodes

def get_init_reward(curChessNode):
    if curChessNode.player == True:
        return -1e4
    else:
        return 1e4

def compute_minimax(root):
    start = time.time()
    for child in root.children:
        root.minimax = max(root.minimax, minimax(child))
    print(f'Computed Minimax values for the tree in {time.time()-start}s')
          
def minimax(curChessNode):
    if len(curChessNode.children) == 0:
        return 1
    elif curChessNode.player == True:
        for child in curChessNode.children:
          curChessNode.minimax = max(curChessNode.minimax, minimax(child)) 
        return curChessNode.minimax
    else:
        for child in curChessNode.children:
          curChessNode.minimax = min(curChessNode.minimax, minimax(child)) 
        return curChessNode.minimax
    
def play_chess(env, curChessNode):
    bestMoveScore = 0
    bestMove = None
    bestAction = None
    
    if len(curChessNode.children) == 0:
        return
    
    elif curChessNode.player == True:
        bestMoveScore = curChessNode.children[0].minimax
        bestMove = curChessNode.children[0]
        bestAction = curChessNode.children[0].action
        
        for child in curChessNode.children:
            if bestMoveScore < child.minimax:
                bestMoveScore = child.minimax
                bestMove = child
                bestAction = child.action
        
    else:
        bestMoveScore = curChessNode.children[0].minimax
        bestMove = curChessNode.children[0]
        bestAction = curChessNode.children[0].action
        
        for child in curChessNode.children:
            if bestMoveScore > child.minimax:
                bestMoveScore = child.minimax
                bestMove = child
                bestAction = child.action
                
    new_state, reward, done, info = env.step(bestAction)
    env.render()
    play_chess(env, bestMove)              
    

if __name__ == "__main__":
    # create the game tree
    env = ChessEnvV2(opponent='none')
    state = env.state
    start = time.time()
    root = ChessNode(state, None, True, 0)
    print(f'Created tree in {time.time()-start}s')
    
    # compute the minimax values
    compute_minimax(root)
    
    # play the game
    #env = ChessEnvV2(opponent='none')
    play_chess(env, root)
    
    