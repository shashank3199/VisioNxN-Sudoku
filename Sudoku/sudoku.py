"""
  sudoku.py             :   This file contains class to solve Sudoku Puzzle.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   3rd September
"""

# method for cartesian product
from itertools import product
# import respective object type for type hint specification
from typing import Dict, List

# import ndarray as type hint specification
from numpy import ndarray


class Sudoku:
    """Template class for finding the solution of the Sudoku Puzzle"""

    def __init__(self, matrix: ndarray, box_row: int = 3, box_col: int = 3):
        """default initialization"""

        # set the matrix 
        self.matrix = matrix
        # save a copy of the initial matrix
        self.init_matrix = self.matrix.copy()
        # set the number of rows in the sub grid
        self.box_row = box_row
        # set the number of columns in the sub grid
        self.box_col = box_col
        # set the game size
        self.N = self.box_row * self.box_col

    def init_row_cols(self):
        """Initialize the rows and columns"""

        # rows for the exact cover problem
        rows = dict()
        for (i, j, n) in product(range(self.N), range(self.N), range(1, self.N + 1)):
            b = (i // self.box_row) * self.box_row + (j // self.box_col)
            rows[(i, j, n)] = [("row-col", (i, j)), ("row-num", (i, n)),
                               ("col-num", (j, n)), ("box-num", (b, n))]

        # cols for the exact cover problem
        columns = dict()
        for (i, j) in product(range(self.N), range(self.N)):
            columns[("row-col", (i, j))] = set()
            columns[("row-num", (i, j + 1))] = set()
            columns[("col-num", (i, j + 1))] = set()
            columns[("box-num", (i, j + 1))] = set()

        for pos, list_value in rows.items():
            for value in list_value:
                columns[value].add(pos)

        return rows, columns

    def solve(self, rows: Dict, cols: Dict, partial_solution: List):
        # if the cols is empty list
        if not cols:
            # yield the part of the solution
            yield list(partial_solution)
        else:
            # select column with min links
            selected_col = min(cols, key=lambda value: len(cols[value]))
            # for each of the value in the selected link
            for values in list(cols[selected_col]):
                # add it to the partial solution considered
                partial_solution.append(values)
                # cover or hide associated links
                removed_cols = self.cover_column(rows, cols, values)
                # recursive call with the values left to cover
                for solution in self.solve(rows, cols, partial_solution):
                    # yield the solution
                    yield solution
                # uncover or unhide associated links
                self.uncover_column(rows, cols, values, removed_cols)
                # remove them from the part of the solution considered
                partial_solution.pop()

    @staticmethod
    def cover_column(rows: Dict, cols: Dict, values):
        """Cover or Hide a column in the exact cover problem"""

        # empty list of removed columns yet
        removed_columns = []
        # for each row in selected column
        for row in rows[values]:
            # for each column in current row
            for row_col in cols[row]:
                # for each row of this column
                for col_row_col in rows[row_col]:
                    # if this row is not the initial row
                    if col_row_col != row:
                        # remove item from set
                        cols[col_row_col].remove(row_col)

            removed_columns.append(cols.pop(row))
        return removed_columns

    @staticmethod
    def uncover_column(rows: Dict, cols: Dict, values, removed_columns: List):
        """Uncover or Unhide a column in the exact cover problem"""

        # since removed columns is stack, work in reverse order of list
        for row in reversed(rows[values]):
            # pop the last column removed
            cols[row] = removed_columns.pop()
            # for row in col of previously deleted row
            for row_col in cols[row]:
                # for col in the above row
                for col_row_col in rows[row_col]:
                    # if this row is not the initial row
                    if col_row_col != row:
                        # add item back to set
                        cols[col_row_col].add(row_col)

    def get_solution(self):
        """ Returns list of possible solutios for the Problem"""

        # initialize rows and columns
        rows, cols = self.init_row_cols()
        # initialize list of solutions
        solutions = []

        # for each row in puzzle matrix
        for i in range(self.N):
            # for each column 
            for j in range(self.N):
                # if the value is not zero
                if self.matrix[i, j] != 0:
                    # remove associated values from the solution space
                    self.cover_column(rows, cols, (i, j, self.matrix[i, j]))

        # iterate through the solutions
        for solution in self.solve(rows, cols, []):
            # iterate through coordinates and there values
            for (i, j, element) in solution:
                # assign the values to the respective elements
                self.matrix[i, j] = element
            # append the solution to the list of solutions
            solutions.append(self.matrix)
            # reset the matrix to the initial matrix
            self.matrix = self.init_matrix.copy()
        # return the list of solutions
        return solutions

    @staticmethod
    def element_possible(matrix: ndarray, box_row: int, box_col: int, i: int, j: int):
        """Helper method to check if a value in the matrix is valid"""

        # backup the element in place
        element = matrix[i, j].copy()
        # reassign as 0
        matrix[i, j] = 0
        # find the sub grid
        sub_r, sub_c = i - i % box_row, j - j % box_col

        not_found = True
        # if element exists in the same row or the same column or the same sub grid
        if element in matrix[i, :] or \
                element in matrix[:, j] or \
                element in matrix[sub_r:sub_r + box_row, sub_c:sub_c + box_col]:
            not_found = False

        # reassign the backup value
        matrix[i, j] = element
        # return the status variable
        return not_found
