"""
  camera_windows.py     :   This file contains the class for implementing a camera screen in pygame.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   3rd September
"""

# import environment variable from os
import os

# import the opencv module to read through the camera
import cv2.cv2 as cv2
# import numpy to appy rotation and flip the image read from the camera
import numpy as np
# import pygame module
import pygame

# button class to create a button to go to main-menu and click the image
from GUI.button import Button

# set the environment variable to open the GUI in the center of the screen
os.environ['SDL_VIDEO_CENTERED'] = '1'

# initialize the pygame module
pygame.init()


class CameraWindow:
    """Template class for creating pygame windows with camera input"""

    def __init__(self, cam_source: int = 0, capture_image: bool = False):
        """default initialization"""

        # default device index of the camera being used
        self.cam_source = cam_source
        # whether to show capture image button
        self.capture_image = capture_image

    def __enter__(self):
        """context manager `__enter__` method invoked on entry to `with` statement."""

        # assign the camera variable
        self.camera = cv2.VideoCapture(self.cam_source)
        # raise error to indicate no web-cam on the current device index
        assert self.camera is not None, "Camera not available, try using a different cam_source value"
        # read the first frame to get the shape of the image
        _, frame = self.camera.read()
        # get frame size
        rows, cols, _ = frame.shape
        # if window meant to capture image
        if self.capture_image:
            # add additional height to the pygame screen for the click button
            self.screen = pygame.display.set_mode((cols, rows + 100))
            # load the icon for the camera button
            self.camera_icon = pygame.image.load('.images/camera_icon.png')
            # add a button to be pressed to click the image
            self.button_click = Button(cols // 2, rows + 50, 250, 60, (200, 200, 200), '  ')
        else:
            # create a screen with the size same as the image
            self.screen = pygame.display.set_mode((cols, rows))
            # load the icon for the home button
            self.home_icon = pygame.image.load('.images/home_icon.png')
            # load a home button to go back to main menu
            self.button_home = Button(60, 60, 70, 70, (200, 200, 200), '  ')
        # return the object to be used on entry to the `with` statement.
        return self

    def get_frame(self):
        """get the current frame from the screen"""

        # read frame from camera
        _, frame = self.camera.read()
        # return frame
        return frame

    def draw_window(self, frame: np.ndarray = None):
        """draw a window the current camera feed"""

        # fill white background
        self.screen.fill((255, 255, 255))
        # if no frame is supplied to draw
        if frame is None:
            # get frame from the camera
            _, frame = self.camera.read()
        # convert frame from BGR format to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # create surface
        pygame_frame = pygame.surfarray.make_surface(np.flip(np.rot90(frame.copy()), axis=0))
        # display surface
        self.screen.blit(pygame_frame, (0, 0))
        # is window needs to click image
        if self.capture_image:
            # display the click button
            self.button_click.draw(self.screen)
            # display the click icon
            self.screen.blit(self.camera_icon,
                             (self.button_click.x - self.camera_icon.get_width() / 2,
                              self.button_click.y - self.camera_icon.get_height() / 2))
        # if window is for live stream
        else:
            # display the home button
            self.button_home.draw(self.screen)
            # display the home icon
            self.screen.blit(self.home_icon,
                             (self.button_home.x - self.home_icon.get_width() / 2,
                              self.button_home.y - self.home_icon.get_height() / 2))
        # update pygame display to show changes
        pygame.display.update()

        # for each event in the game
        for event in pygame.event.get():
            # kill window, if game window is closed
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # if click button is pressed 
            elif self.capture_image and self.button_click.clicked(event):
                # return the current image frame
                return frame

            # if home button is pressed
            elif not self.capture_image and self.button_home.clicked(event):
                # return to main menu
                return False
        # return None by default
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """context manager `__exit__` method invoked during exit from `with` statement."""

        # release the camera variable object
        self.camera.release()
