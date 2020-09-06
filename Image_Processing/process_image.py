"""
  process_image.py      :   This file contains the class for recognition of the sudoku from image.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   4th September
"""

# import the opencv module for image processing
import cv2.cv2 as cv2
# import numpy module for operations on image matrices
import numpy as np
# import torch module
import torch

# import load_model method in order to classify the images
from Image_Processing.classifier import load_model

# name of model file
model_file = './Image_Processing/char74k-cnn.pth'


class SudokuImageProcessing:
    """
    Template class for image processing to recognise the puzzle from the image 
    and plot to plot back the solution in case of augmented reality.
    """

    def __init__(self, image: np.ndarray = None, fname: str = None):
        """default initialization"""

        # if image not provided
        if image is None:
            # check for path name
            assert fname is not None, "file name not entered"
            # read image
            self.image = cv2.imread(fname)
            # if image is invalid
            assert self.image is not None, "unable to open file: {}".format(fname)

        # if image provided
        else:
            # set image for the class
            self.image = image

        # initialize default size as None
        self.game_size = None
        # initialize default box rows as None
        self.box_rows = None
        # initialize default box cols as None
        self.box_cols = None

    def get_grid(self):
        """
        Detect the outter grid of the sudoku and return the area inside this grid.
        """

        # convert the image to gray scale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # apply gaussian threshold
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        # find all the contours
        contours = cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        # select the largest contour
        max_cnt = max(contours, key=cv2.contourArea)

        # check if the contour area is greater than 250 ** 2 pixel sq.
        if cv2.contourArea(max_cnt) < 250 * 250:
            # if not, then return None
            return None, None

        # create black mask
        mask = np.zeros(gray.shape, np.uint8)
        # fill the area inside contour with white pixels
        cv2.drawContours(mask, [max_cnt], 0, 255, -1)
        # fill the rest area with black pixels
        cv2.drawContours(mask, [max_cnt], 0, 0, 2)

        # create a white mask
        out = 255 * np.ones_like(gray)
        # copy the area from the original image which is marked by white pixels from the black mask.
        out[mask == 255] = gray[mask == 255]
        # return the generated white mask and largest contour
        return out, max_cnt

    def get_warped(self):
        """
        Apply warp perspective tranformation by finding the corners of the contour to convert the
        sudoku from being part of the image to the complete image itself.
        """

        # get the generated white mask and the largest contour
        img, max_cnt = self.get_grid()
        # if the contour detected had an area smaller than 250 ** 2 pixels sq.
        if img is None:
            # return None 
            return None, None

        # calculte the perimeter of the contour
        peri = cv2.arcLength(max_cnt, True)
        # approximates the polygonal curves to detect vertices
        approx = cv2.approxPolyDP(max_cnt, 0.015 * peri, True)
        # flatten the vertices array
        pts = np.squeeze(approx)
        # find width of the puzzle
        box_width = np.max(pts[:, 0]) - np.min(pts[:, 0])
        # find height of the puzzle
        box_height = np.max(pts[:, 1]) - np.min(pts[:, 1])

        """
        The following steps are used to approximate the corner coordinates of the puzzle
        in order to apply an appropriate transformation.
        """

        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1)
        bounding_rect = np.array([pts[np.argmin(sum_pts)],
                                  pts[np.argmin(diff_pts)],
                                  pts[np.argmax(sum_pts)],
                                  pts[np.argmax(diff_pts)]], dtype=np.float32)

        dst = np.array([[0, 0],
                        [box_width - 1, 0],
                        [box_width - 1, box_height - 1],
                        [0, box_height - 1]], dtype=np.float32)

        # generate the transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(bounding_rect, dst)
        # apply the tranformation matrix to get the primary sudoku image
        warped_img = cv2.warpPerspective(img, transform_matrix, (box_width, box_height))

        # return the warped_img and its tranformation matrix
        return warped_img, transform_matrix

    def get_dimensions(self):
        """
        Get the dimensions of the sudoku, the aim of this function is to detect out of the following -

                game_size    |    game_dim    
            ----------------------------------
                    4        |      2 X 2
                    6        |      2 X 3
                    6        |      3 X 2
                    8        |      2 X 4
                    8        |      4 X 2
                    9        |      3 X 3

        Higher dimensions can be also detected, provided the camera has better resolution.

        Note: The approach specified here will only work when a thicker border is used to 
              distinguish between the sub-grids of the puzzle.
        """

        # get the warped image
        img, _ = self.get_warped()
        # if no image is received
        if img is None:
            # return None
            return None, (None, None)

        # apply gaussian blur
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        # apply thresholding such that only the thicker border is visible
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        """
        Note: The image now contains only 0 or 255 as value for any pixel
        """

        # mask the surrounding border of the puzzle
        # top side
        thresh[0:10, :] = 255
        # left side
        thresh[:, 0:10] = 255
        # bottom side
        thresh[-10:, :] = 255
        # right side
        thresh[:, -10:] = 255
        # generate the inverse of the image and map it to 0 or 1
        thresh = np.bitwise_not(thresh) / 255

        # get sum of each column
        vertical_sum = np.sum(np.round(thresh, 0), axis=0)
        # get sum of each row
        horizontal_sum = np.sum(np.round(thresh, 0), axis=1)

        # set the minimum height as 2/3rd of the complete image height
        min_height = int(0.66 * img.shape[0])
        # set the minimum width as 2/3rd of the complete image width
        min_width = int(0.66 * img.shape[1])

        """
        
                             Image           Horizontal Bool

                00000000101000000000110000000000    0
                01000000101001000000101010000000    0
                00010000111000000000111000100000    0
                00000100111010001000110000001000    0
                00000000110000000000100000000000    0
                11111000010111111110101111111011    1
                00001011111100000011111100011110    1
                00000000110000000000111000100000    0
                01000000011001000000011000000000    0
                00010000101000010000110010000000    0
                00000100111000000000110000000000    0
                11111100110111111111111111111011    1
                00011111111111111111110001111111    1
                00000000011000000000011000000000    0
                00001000011000100000111001001000    0
                00000000101000000000110000000000    0
                00000000111000000000011000000000    0
                

                00000000111000000000111000000000

                        Vertical Bool


        """
        vertical_bool = vertical_sum > min_height
        horizontal_bool = horizontal_sum > min_width

        # detects the total number of `01` patterns in the vertical bool
        v_lines = 1
        for i in range(1, len(vertical_bool)):
            v_lines += (~ vertical_bool[i - 1]) & vertical_bool[i]

        # detects the total number of `01` patterns in the horizontal bool
        h_lines = 1
        for i in range(1, len(horizontal_bool)):
            h_lines += (~ horizontal_bool[i - 1]) & horizontal_bool[i]

        """
        v_lines corresponds to the total number of rows in a sub grid
        h_lines corresponds to the total number of columns in a sub grid
        """

        # the game size is the product of the sub_grid dimensions
        game_size = v_lines * h_lines

        """
        Modify the line below in order to add other valid dimensions if possible.
        """
        # since the camera can clearly detect only sizes 4,6,8,9
        if game_size not in (4, 6, 8, 9):
            # raise exception for improper grid size
            raise RuntimeError("Improper Grid Size, expected in {{4,6,8,9}}, got {}".format(game_size))

        # return the game dimensions
        return game_size, (v_lines, h_lines)

    @staticmethod
    def preprocess_digit(digit_img):
        """
        Helper method to clear borders and darken the digit of an individual cell of the puzzle.
        """

        # expand the image
        digit_img = cv2.resize(digit_img, (112, 112),
                               interpolation=cv2.INTER_CUBIC)
        # apply gaussian blur
        digit_img = cv2.GaussianBlur(digit_img, (5, 5), 0)
        # apply threshold to the image
        digit_img = cv2.threshold(digit_img, 135, 255, cv2.THRESH_TRUNC)[1]
        # set all pixels with value greater than 110 as 255
        digit_img[digit_img >= 110] = 255

        # mask the surrounding border of the cell
        # top side
        digit_img[0:10, :] = 255
        # left side
        digit_img[:, 0:10] = 255
        # bottom side
        digit_img[-10:, :] = 255
        # right side
        digit_img[:, -10:] = 255

        # resize image
        digit_img = cv2.resize(digit_img, (28, 28),
                               interpolation=cv2.INTER_CUBIC)

        # if there are less that 10 black pixels
        if np.sum(np.bitwise_not(digit_img)) < 255 * 10:
            return None

        # scale pixels with value less than 150 to 3/4th the value
        pos = digit_img < 150
        digit_img[pos] = 3 * (digit_img[pos] // 4)
        # double the values of other pixels
        digit_img[np.bitwise_not(pos)] = 2 * digit_img[np.bitwise_not(pos)]

        # return the cell image
        return digit_img

    def get_matrix(self):
        """Returns the puzzle matrix from the image"""

        # get the warped image
        img, _ = self.get_warped()

        # if warped image is None
        if img is None:
            # return None
            return None

        # get the dimensions for the game
        self.game_size, (self.box_rows, self.box_cols) = self.get_dimensions()

        # initialize matrix with zeros
        matrix = np.zeros((self.game_size, self.game_size), dtype=int)
        # load the lassifier model
        model = load_model(model_file)
        # set the model to evaluation mode, i.e. now the imputs will individual not in batches
        model.eval()

        # get the height and width of each cube or cell
        cube_h, cube_w = np.array(img.shape) / self.game_size

        # iterate through the rows
        for i in range(self.game_size):
            # get start pixel height
            y_start = int(i * cube_h)
            # get end pixel height
            y_end = int((i + 1) * cube_h)

            # iterate through columns
            for j in range(self.game_size):
                # get start pixel width
                x_start = int(j * cube_w)
                # get end pixel width
                x_end = int((j + 1) * cube_w)

                # copy the cell
                digit_img = img[y_start:y_end, x_start:x_end].copy()
                # preprocess the cell
                digit_img = self.preprocess_digit(digit_img)
                # if the preprocessing returns None
                if digit_img is None:
                    # set value as 0 
                    matrix[i, j] = 0
                    # continue with next cell
                    continue

                # map the image pixel values between 0-1
                digit_img = digit_img / 255
                # reshape the numpy array
                digit_img = np.array(digit_img).reshape((1, 1, 28, 28))
                # convert the image to tensor
                digit_img_tensor = torch.tensor(digit_img, dtype=torch.float)
                # get the model prediction
                digit_img_out = np.array(model(digit_img_tensor).detach(), dtype=np.float32).flatten()
                # digit is the index of the max value in predicted outputs
                element = int(np.argmax(digit_img_out))

                """
                Sanity Check: This will ensure that an impossible puzzle does not get loaded because
                              2 cells in a row or in a column or in a sub grid have the same value.

                              If they have same values, then the one with higher prediction score for 
                              the specific label gets the value of the element and the other variable 
                              chooses the value with the second highest prediction score.
                """

                # sub grid position
                sub_r, sub_c = i - (i % self.box_rows), j - (j % self.box_cols)
                # sub grid values
                sub_matrix = matrix[sub_r:sub_r + self.box_rows, sub_c:sub_c + self.box_cols]

                # if value in the same row
                if element in matrix[i, :]:
                    x = i
                    y = list(matrix[i, :]).index(element)

                # if the value in the same column
                elif element in matrix[:, j]:
                    x = i
                    y = list(matrix[:, j]).index(element)

                # if the value in the same sub grid
                elif element in sub_matrix:
                    pos = list(sub_matrix.flatten()).index(element)
                    x = sub_r + pos // self.box_rows
                    y = sub_c + pos % self.box_cols

                # value not found any where else
                else:
                    matrix[i, j] = element
                    continue

                # get the image of the duplicate cell
                duplicate = np.copy(img[int(x * cube_h):int((x + 1) * cube_h),
                                    int(y * cube_w):int((y + 1) * cube_w)])
                # preprocess the cell
                duplicate = self.preprocess_digit(duplicate)
                # if the preprocessing returns None
                if duplicate is None:
                    # set the current index with the current element value
                    matrix[i, j] = element
                    # set the duplicate cell with 0
                    matrix[x, y] = 0
                    continue

                # map the image pixel values between 0-1
                duplicate = duplicate / 255
                # reshape the numpy array
                duplicate = np.array(duplicate).reshape((1, 1, 28, 28))
                # convert the image to tensor
                duplicate_tensor = torch.tensor(duplicate, dtype=torch.float)
                # get the model prediction
                duplicate_out = np.array(model(duplicate_tensor).detach(), dtype=np.float32).flatten()

                # if prediction score of current element is more
                if digit_img_out[element] > duplicate_out[element]:
                    # set the current index with the current element value
                    matrix[i, j] = element
                    # assign the index in duplicate output with negative infinity
                    duplicate_out[element] = np.NINF
                    # set the duplicate index with the new highest of its respective output
                    matrix[x, y] = int(np.argmax(duplicate_out))

                # if prediction score of duplicate element is more
                else:
                    # set the duplicate index with the current element value
                    matrix[x, y] = element
                    # assign the index in current output with negative infinity
                    digit_img_out[element] = np.NINF
                    # set the current index with the new highest of its respective output
                    matrix[i, j] = int(np.argmax(digit_img_out))

        # return the matrix
        return matrix

    @staticmethod
    def plot_on_image(image, matrix, matrix_sol, game_dim):
        """Plot the answer on the initial image"""

        # if the detected matrix or the solution for it is none
        if matrix is None or matrix_sol is None:
            # return the original image without any changes
            return image

        # get size of the puzzle
        game_size = np.prod(game_dim)

        # convert the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # apply gaussian threshold
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        # find all the contours
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        # select the largest contour
        max_cnt = max(contours, key=cv2.contourArea)

        # calculte the perimeter of the contour
        peri = cv2.arcLength(max_cnt, True)
        # approximates the polygonal curves to detect vertices
        approx = cv2.approxPolyDP(max_cnt, 0.015 * peri, True)
        # flatten the vertices array
        pts = np.squeeze(approx)
        # find width of the puzzle
        box_width = np.max(pts[:, 0]) - np.min(pts[:, 0])
        # find height of the puzzle
        box_height = np.max(pts[:, 1]) - np.min(pts[:, 1])

        """
        The following steps are used to approximate the corner coordinates of the puzzle
        in order to apply an appropriate transformation.
        """

        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1)
        bounding_rect = np.array([pts[np.argmin(sum_pts)],
                                  pts[np.argmin(diff_pts)],
                                  pts[np.argmax(sum_pts)],
                                  pts[np.argmax(diff_pts)]], dtype=np.float32)

        dst = np.array([[0, 0],
                        [box_width - 1, 0],
                        [box_width - 1, box_height - 1],
                        [0, box_height - 1]], dtype=np.float32)

        # generate the transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(bounding_rect, dst)
        # apply the tranformation matrix to get the primary sudoku image
        warped_img = cv2.warpPerspective(image, transform_matrix, (box_width, box_height))

        # get the height and width of each cube or cell
        cube_h, cube_w, _ = np.array(warped_img.shape) / game_size
        # generate mask on which the numbers will be drawn
        template = np.zeros(warped_img.shape)

        # iterate through rows
        for i in range(game_size):
            # get start pixel height
            y_start = int(i * cube_h) + int(cube_h / 1.25)

            # iterate throught columns
            for j in range(game_size):
                # get start pixel width
                x_start = int(j * cube_w) + int(cube_w / 4)

                # if its an empty cell
                if matrix[i, j] == 0:
                    """
                    font_size = (14 - game_size)/5;
                    
                    this is an equation found by experimenting with various font sizes on
                    different game sizes and settling for the best fit.
                    """

                    # add text on the mask
                    cv2.putText(template, str(matrix_sol[i, j]), (x_start, y_start),
                                cv2.FONT_HERSHEY_SIMPLEX, (14 - game_size) / 5,
                                (255, 255, 255), 3, cv2.LINE_AA)

        # get the height and width of the image on which the result is to be plotted
        img_height, img_width, _ = image.shape
        try:
            # apply the inverse of the transform which was applied to bring the sudoku to frame
            unwarped_mask = cv2.warpPerspective(template,
                                                np.linalg.inv(transform_matrix),
                                                (img_width, img_height))
        # incase the transformation matrix is Singular            
        except np.linalg.LinAlgError:
            # return original image
            return image
        # generate a copy of original image
        image_masked = image.copy()
        # on this plot the mask with the predicted solution
        image_masked[unwarped_mask == 255] = 0
        # return the image with the solution
        return image_masked
