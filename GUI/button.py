"""
  button.py             :   This file contains the class for implementing a pygame button.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   3rd September
"""

# import respective object type for type hint specification
from typing import Tuple

# import pygame module
import pygame

# initialize pygame fonts
pygame.font.init()


class Button:
    """Template class to implement a mouse-based button in pygame GUI"""

    def __init__(self, x: int, y: int, button_width: int, button_height: int, color: Tuple, text: str):
        """default initialization"""

        # central x-coordinate
        self.x = x
        # central y-coordinate
        self.y = y
        # set width of button
        self.button_width = button_width
        # set height of button
        self.button_height = button_height
        # set fill color of the button
        self.color = color
        # color of button when mouse hovers over it, it is 75% of the fill color
        self.hover_color = tuple(3 * (c // 4) for c in color)
        # radius of curvature of the button
        self.radius = 0.5
        # set text to be displayed on the button
        self.text = text

    def draw(self, surface: pygame.Surface):
        """used to draw the button on a surface"""

        # create the outline rectangle for the button
        rect = pygame.Rect(self.x - self.button_width // 2,
                           self.y - self.button_height // 2,
                           self.button_width,
                           self.button_height)

        # check if current mouse position is over the button area
        if self.under_mouse():
            # set fill color
            color = pygame.Color(*self.hover_color)
        else:
            # otherwise darken the button
            color = pygame.Color(*self.color)
        # specifies opacity of the color
        alpha = color.a
        # alpha componenent of pygame Color object
        color.a = 0

        # save the top-left coordinate of the rectangle
        pos = rect.topleft
        # re-assign the top-left coordinate of the rectangle as (0, 0)
        rect.topleft = 0, 0

        # create a rectangular surface , SRCALPHA implies pixel format will include per-pixel alpha
        rectangle = pygame.Surface(rect.size, pygame.SRCALPHA)
        # create a circular surface for rounded borders
        circle = pygame.Surface([min(rect.size) * 3] * 2, pygame.SRCALPHA)
        # draw a circle on the circle surface
        pygame.draw.ellipse(circle, (0, 0, 0), circle.get_rect(), 0)
        # scale the circle
        circle = pygame.transform.smoothscale(circle, [int(min(rect.size) * self.radius)] * 2)

        # draw top-left circle on the rectangle surface
        radius = rectangle.blit(circle, (0, 0))
        # draw bottom-right circle on the rectangle surface
        radius.bottomright = rect.bottomright
        rectangle.blit(circle, radius)
        # draw top-right circle on the rectangle surface
        radius.topright = rect.topright
        rectangle.blit(circle, radius)
        # draw bottom-left circle on the rectangle surface
        radius.bottomleft = rect.bottomleft
        rectangle.blit(circle, radius)

        # fill the complete button with the color
        rectangle.fill((0, 0, 0), rect.inflate(-radius.w, 0))
        rectangle.fill((0, 0, 0), rect.inflate(0, -radius.h))
        rectangle.fill(color, special_flags=pygame.BLEND_RGBA_MAX)
        rectangle.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MIN)

        # make label for the text
        font = pygame.font.SysFont('comicsans', self.button_height // 2)
        label = font.render(self.text, 1, (0, 0, 0))

        # display the button the respective surface
        surface.blit(rectangle, pos)
        # display the text over the button
        surface.blit(label, (self.x - label.get_width() // 2,
                             self.y - label.get_height() // 2))

    def clicked(self, event):
        """used to check if the button is clicked"""

        # if LEFT mouse button is clicked
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # if the mouse was over the button during click
            return self.under_mouse()

    def under_mouse(self):
        """find if the current mouse coordinates"""

        # get the current coordinates of the mouse
        mouse_x, mouse_y = pygame.mouse.get_pos()
        # if mouse coordinates during the click are in the range of the button coordinate
        if mouse_x in range(self.x - self.button_width // 2, self.x + self.button_width // 2) and \
                mouse_y in range(self.y - self.button_height // 2, self.y + self.button_height // 2):
            # return true
            return True
        # otherwise return false
        return False
