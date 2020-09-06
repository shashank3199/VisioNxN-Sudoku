"""
  main.py               :   This file contains the game and its working.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   3rd September
"""

# import exists to check if a last loaded file exists in the directory
from os.path import exists

# import numpy module to load the laste loaded puzzle or the sample
import numpy as np
# import pygame module
import pygame

# import the main GUI for the program
from game_window import SudokuGUI

# initialize the pygame runtime
pygame.init()
# set caption for game window
pygame.display.set_caption('VisioNxN Sudoku')
# load icon for game
icon = pygame.image.load('./.images/game_logo.png')
# set icon for the game
pygame.display.set_icon(icon)


def main_game():
    """main method to be called"""

    # infinte loop
    while True:
        # if last loaded puzzle exists
        if exists('last_loaded.npy') and exists('last_loaded_dim.npy'):
            # load the last loaded puzzle
            mat = np.load('last_loaded.npy')
            # load the last loaded dimensions
            box_rows, box_cols = tuple(np.load('last_loaded_dim.npy'))
        else:
            # load the sample puzzle
            mat = np.load('sample.npy')
            # load the sample puzzle dimensions
            box_rows, box_cols = (3, 3)
        # creat an instance of the GUI for the game
        sg = SudokuGUI(mat.copy(), box_rows, box_cols)

        # initial value for the menu option
        menu_val = 0
        # run while the main_menu doesnt generate a valid option
        while not menu_val:
            menu_val = sg.main_menu()

        # if play game option is selected
        if menu_val == 1:
            # play game while home button is not clicked
            while sg.play_game():
                pass
        # if augmented reality option is selected
        else:
            # load augmented reality screen until home button is not clicked
            while sg.load_AR():
                pass


if __name__ == '__main__':
    main_game()
