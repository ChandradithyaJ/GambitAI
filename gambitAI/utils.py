import os

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
    
def get_num_pieces_on_board(board):
    num_pieces = 0
    for row in board:
        for piece in row:
            if piece != 0:
                num_pieces += 1
    return num_pieces