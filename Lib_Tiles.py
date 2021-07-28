import numpy as np


def return_CD09_ORCA_Tiles():
    result = None
    matrix_left_edge = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]])
    matrix_middle = np.array(
        [[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    result = [matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_middle,
              matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge,
              matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_middle,
              matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge]
    return result


def return_CD09_506_Tiles():
    result = None
    matrix_left_edge = np.array(
        [[0, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
    matrix_middle = np.array(
        [[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
    result = [matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_middle,
              matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge,
              matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_middle,
              matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge]
    return result


def return_CD11_Tiles():
    result = None
    matrix = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    result = [matrix] * 72
    return result


def return_CD13_Tiles():
    result = None
    matrix = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    result = [matrix] * 96
    return result


def return_CD21_Tiles():
    result = None
    matrix = np.array(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    result = [matrix] * 96
    return result


def return_CD22_Tiles():
    result = None
    matrix = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    result = [matrix] * 96
    return result


def return_CD23_Tiles():
    result = None
    matrix = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    result = [matrix] * 96
    return result


def return_CD24_Tiles():
    result = None
    matrix = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    result = [matrix] * 96
    return result


def return_96well_25_Tiles():
    result = None
    matrix = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    result = [matrix] * 96
    return result


def return_12well_324_Tiles():
    result = None
    matrix = np.array(
        [[0] * 5 + [1] * 8 + [0] * 5, [0] * 3 + [1] * 12 + [0] * 3, [0] * 2 + [1] * 14 + [0] * 2, [0] + [1] * 16 + [0],
         [0] + [1] * 16 + [0], [1] * 18, [1] * 18, [1] * 18, [1] * 18, [1] * 18, [1] * 18, [1] * 18, [1] * 18,
         [0] + [1] * 16 + [0], [0] + [1] * 16 + [0], [0] * 2 + [1] * 14 + [0] * 2, [0] * 3 + [1] * 12 + [0] * 3,
         [0] * 5 + [1] * 8 + [0] * 5])
    result = [matrix] * 12
    return result


def return_24well_Tiles():
    result = None
    matrix_left_edge = np.array(
        [[1] * 7 + [0] * 3, [1] * 8 + [0] * 2, [1] * 9 + [0] * 1, [1] * 10, [1] * 10, [1] * 10, [1] * 10, [1] * 10,
         [1] * 10, [1] * 10, [1] * 9 + [0] * 1, [1] * 8 + [0] * 2, [1] * 7 + [0] * 3])
    matrix_right_edge = np.array(
        [[0] * 3 + [1] * 7 + [0] * 2, [0] * 2 + [1] * 9 + [0] * 1, [0] * 1 + [1] * 11, [1] * 12, [1] * 12, [1] * 12,
         [1] * 12, [1] * 12, [1] * 12, [1] * 12, [0] * 1 + [1] * 11, [0] * 2 + [1] * 9 + [0] * 1,
         [0] * 3 + [1] * 7 + [0] * 2])
    matrix_middle = np.array(
        [[0] * 3 + [1] * 7 + [0] * 3, [0] * 2 + [1] * 9 + [0] * 2, [0] * 1 + [1] * 11 + [0] * 1, [1] * 13, [1] * 13,
         [1] * 13, [1] * 13, [1] * 13, [1] * 13, [1] * 13, [0] * 1 + [1] * 11 + [0] * 1, [0] * 2 + [1] * 9 + [0] * 2,
         [0] * 3 + [1] * 7 + [0] * 3])
    result = [matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_right_edge,
              matrix_right_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge,
              matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_right_edge,
              matrix_right_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge]
    return result


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Lib_Tiles.py !')
