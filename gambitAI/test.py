import random
from gym_chess import ChessEnvV2

env = ChessEnvV2(opponent='none')
state = env.state
moves = env.possible_moves
move = random.choice(moves)
print("Move: ", move)
action = env.move_to_action(move)

new_state, reward, done, info = env.step(action)

env.render()

state = env.reset()
