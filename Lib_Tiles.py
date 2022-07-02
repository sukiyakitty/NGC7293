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


def return_384well_9_Tiles():
    result = None
    matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    result = [matrix] * 384
    return result


def return_12well_324_Tiles():
    result = None
    matrix = np.array(
        [[0] * 5 + [1] * 8 + [0] * 5,
         [0] * 3 + [1] * 12 + [0] * 3,
         [0] * 2 + [1] * 14 + [0] * 2,
         [0] + [1] * 16 + [0],
         [0] + [1] * 16 + [0],
         [1] * 18,
         [1] * 18,
         [1] * 18,
         [1] * 18,
         [1] * 18,
         [1] * 18,
         [1] * 18,
         [1] * 18,
         [0] + [1] * 16 + [0],
         [0] + [1] * 16 + [0],
         [0] * 2 + [1] * 14 + [0] * 2,
         [0] * 3 + [1] * 12 + [0] * 3,
         [0] * 5 + [1] * 8 + [0] * 5])
    result = [matrix] * 12
    return result


def return_24well_3344_Tiles():
    # 3344 tiles
    result = None
    matrix_left_edge = np.array(
        [[1] * 7 + [0] * 3,
         [1] * 8 + [0] * 2,
         [1] * 9 + [0] * 1,
         [1] * 10,
         [1] * 10,
         [1] * 10,
         [1] * 10,
         [1] * 10,
         [1] * 10,
         [1] * 10,
         [1] * 9 + [0] * 1,
         [1] * 8 + [0] * 2,
         [1] * 7 + [0] * 3])
    matrix_right_edge = np.array(
        [[0] * 3 + [1] * 7 + [0] * 2,
         [0] * 2 + [1] * 9 + [0] * 1,
         [0] * 1 + [1] * 11,
         [1] * 12,
         [1] * 12,
         [1] * 12,
         [1] * 12,
         [1] * 12,
         [1] * 12,
         [1] * 12,
         [0] * 1 + [1] * 11,
         [0] * 2 + [1] * 9 + [0] * 1,
         [0] * 3 + [1] * 7 + [0] * 2])
    matrix_middle = np.array(
        [[0] * 3 + [1] * 7 + [0] * 3,
         [0] * 2 + [1] * 9 + [0] * 2,
         [0] * 1 + [1] * 11 + [0] * 1,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [0] * 1 + [1] * 11 + [0] * 1,
         [0] * 2 + [1] * 9 + [0] * 2,
         [0] * 3 + [1] * 7 + [0] * 3])
    result = [matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_right_edge,
              matrix_right_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge,
              matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_right_edge,
              matrix_right_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge]
    return result


def return_24well_3432_Tiles():
    # 3432 tiles
    result = None
    matrix_left_edge = np.array(
        [[0] * 1 + [1] * 7 + [0] * 3,
         [1] * 10 + [0] * 1,
         [1] * 10 + [0] * 1,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [1] * 10 + [0] * 1,
         [1] * 10 + [0] * 1,
         [0] * 1 + [1] * 7 + [0] * 3])
    matrix_middle = np.array(
        [[0] * 3 + [1] * 7 + [0] * 3,
         [0] * 1 + [1] * 11 + [0] * 1,
         [0] * 1 + [1] * 11 + [0] * 1,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [1] * 13,
         [0] * 1 + [1] * 11 + [0] * 1,
         [0] * 1 + [1] * 11 + [0] * 1,
         [0] * 3 + [1] * 7 + [0] * 3])
    matrix_right_edge = np.array(
        [[0] * 3 + [1] * 7 + [0] * 1,
         [0] * 1 + [1] * 10,
         [0] * 1 + [1] * 10,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [1] * 11,
         [0] * 1 + [1] * 10,
         [0] * 1 + [1] * 10,
         [0] * 3 + [1] * 7 + [0] * 1])
    result = [matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_right_edge,
              matrix_right_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge,
              matrix_left_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_right_edge,
              matrix_right_edge, matrix_middle, matrix_middle, matrix_middle, matrix_middle, matrix_left_edge]
    return result


def return_6well_4126_Tiles():
    # 4126 tiles
    result = None
    matrix_left_edge = np.array(
        [[0] * 7 + [1] * 11 + [0] * 9,
         [0] * 5 + [1] * 15 + [0] * 7,
         [0] * 3 + [1] * 19 + [0] * 5,
         [0] * 2 + [1] * 21 + [0] * 4,
         [0] * 1 + [1] * 23 + [0] * 3,
         [1] * 25 + [0] * 2,
         [1] * 25 + [0] * 2,
         [1] * 26 + [0] * 1,
         [1] * 26 + [0] * 1,
         [1] * 27, [1] * 27, [1] * 27,
         [1] * 27, [1] * 27, [1] * 27,
         [1] * 27, [1] * 27, [1] * 27,
         [1] * 27, [1] * 27,
         [1] * 26 + [0] * 1,
         [1] * 26 + [0] * 1,
         [1] * 25 + [0] * 2,
         [1] * 25 + [0] * 2,
         [0] * 1 + [1] * 23 + [0] * 3,
         [0] * 2 + [1] * 21 + [0] * 4,
         [0] * 3 + [1] * 19 + [0] * 5,
         [0] * 5 + [1] * 15 + [0] * 7,
         [0] * 7 + [1] * 11 + [0] * 9])
    matrix_middle = np.array(
        [[0] * 9 + [1] * 11 + [0] * 9,
         [0] * 7 + [1] * 15 + [0] * 7,
         [0] * 5 + [1] * 19 + [0] * 5,
         [0] * 4 + [1] * 21 + [0] * 4,
         [0] * 3 + [1] * 23 + [0] * 3,
         [0] * 2 + [1] * 25 + [0] * 2,
         [0] * 2 + [1] * 25 + [0] * 2,
         [0] * 1 + [1] * 27 + [0] * 1,
         [0] * 1 + [1] * 27 + [0] * 1,
         [1] * 29, [1] * 29, [1] * 29,
         [1] * 29, [1] * 29, [1] * 29,
         [1] * 29, [1] * 29, [1] * 29,
         [1] * 29, [1] * 29,
         [0] * 1 + [1] * 27 + [0] * 1,
         [0] * 1 + [1] * 27 + [0] * 1,
         [0] * 2 + [1] * 25 + [0] * 2,
         [0] * 2 + [1] * 25 + [0] * 2,
         [0] * 3 + [1] * 23 + [0] * 3,
         [0] * 4 + [1] * 21 + [0] * 4,
         [0] * 5 + [1] * 19 + [0] * 5,
         [0] * 7 + [1] * 15 + [0] * 7,
         [0] * 9 + [1] * 11 + [0] * 9])
    matrix_right_edge = np.array(
        [[0] * 9 + [1] * 11 + [0] * 7,
         [0] * 7 + [1] * 15 + [0] * 5,
         [0] * 5 + [1] * 19 + [0] * 3,
         [0] * 4 + [1] * 21 + [0] * 2,
         [0] * 3 + [1] * 23 + [0] * 1,
         [0] * 2 + [1] * 25,
         [0] * 2 + [1] * 25,
         [0] * 1 + [1] * 26,
         [0] * 1 + [1] * 26,
         [1] * 27, [1] * 27, [1] * 27,
         [1] * 27, [1] * 27, [1] * 27,
         [1] * 27, [1] * 27, [1] * 27,
         [1] * 27, [1] * 27,
         [0] * 1 + [1] * 26,
         [0] * 1 + [1] * 26,
         [0] * 2 + [1] * 25,
         [0] * 2 + [1] * 25,
         [0] * 3 + [1] * 23 + [0] * 1,
         [0] * 4 + [1] * 21 + [0] * 2,
         [0] * 5 + [1] * 19 + [0] * 3,
         [0] * 7 + [1] * 15 + [0] * 5,
         [0] * 9 + [1] * 11 + [0] * 7])
    result = [matrix_left_edge, matrix_middle, matrix_right_edge,
              matrix_right_edge, matrix_middle, matrix_left_edge]
    return result


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Lib_Tiles.py !')
