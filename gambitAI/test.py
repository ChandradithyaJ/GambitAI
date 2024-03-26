from gym_chess import ChessEnvV2

K_PLY = 3

class ChessNode:
    def __init__(self, state, minimax, player, depth):
        self.state = state # env state
        self.minimax = minimax # float value
        self.player = player # boolean, WHITE - TRUE, BLACK - FALSE
        self.depth = depth
        
    @property
    def children(self):
        childChessNodes = []
        
        if self.depth < K_PLY:
            env = ChessEnvV2(opponent='none', initial_board=self.state['board'])
            moves = env.possible_moves
            actions = [env.move_to_action(move) for move in moves]
            for action in actions:
                new_state, reward, done, info = env.step(action)
                childChessNodes.append(ChessNode(new_state, 1, not self.player, self.depth + 1))
                initial_state = env.reset()
            
        return childChessNodes
     
