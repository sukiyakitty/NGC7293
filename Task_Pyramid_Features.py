import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import cv2
from Lib_Class import ImageData


def save_image_features_csv(main_path, output_name, index, content, result_path='Pyramid'):
    # output_name must be a csv format: 'example.csv'
    # content must be a python list format: [1,2,3,4,...]

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    result_path = os.path.join(main_path, result_path)
    if os.path.exists(result_path):
        # shutil.rmtree(output_folder)
        pass
    else:
        os.makedirs(result_path)  # make the output folder
    if type(index) is not str:
        print('!ERROR! index must be str!')
        return False
    if len(content) < 1:
        print('!ERROR! Empty content!')
        return False

    Scsv_file = os.path.join(main_path, result_path, output_name)
    if not os.path.exists(Scsv_file):
        Scsv_mem = pd.DataFrame(columns=['f' + str(col) for col in range(1, 1 + len(content))])
    else:
        Scsv_mem = pd.read_csv(Scsv_file, header=0, index_col=0)

    Scsv_mem.loc[index] = np.hstack(content)
    Scsv_mem.to_csv(path_or_buf=Scsv_file)
    return True


def pyramid_features(main_path, input_img, name_index, result_path='Pyramid'):
    # image Pyramid Features extraction
    # input is numpy image or input_img =r'S:\DeskTop\CD22\SSSS_100%\S2\2019-05-18~IPS-1_CD22~T1.png'
    # outout is Pyramid Features CSV file

    debug = False
    check_density = 0.11

    # p_2_2 and p_3_3 is bloc list ::: (row index number, col index number)
    p_2_2 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    p_3_3 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    check_times = 4
    check_list = -np.ones((check_times, 4), dtype=np.int32)
    for i in range(0, check_times):
        check_list[i, 0] = random.randint(0, 9 - 1)
        check_list[i, 1] = random.randint(0, 4 - 1)
        check_list[i, 2] = random.randint(0, 4 - 1)
        check_list[i, 3] = random.randint(0, 4 - 1)

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if type(input_img) is str:
        input_img = os.path.join(main_path, input_img)
        if not os.path.exists(input_img):
            print('!ERROR! The image path does not existed!')
            return False
        t_path_2list = os.path.split(input_img)  # [r'D:\pro\CD22\SSSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
        S_index = os.path.split(t_path_2list[0])[1]  # this well name  'S1'
        name_index = t_path_2list[1][:-4]  # this name (include time point ) but no 'S1~'  '2018-11-28~IPS_CD13~T1'
    elif type(input_img) is np.ndarray:
        pass
    else:
        print('!ERROR! Please input correct CV2 image file or file path!')
        return False

    if name_index is None or name_index == '':
        print('!ERROR! if input CV2 Numpy image format, you must input a index for this image!')
        return False

    result_path = os.path.join(main_path, result_path)
    if os.path.exists(result_path):
        # shutil.rmtree(output_folder)
        pass
    else:
        os.makedirs(result_path)  # make the output folder

    all_features = []
    img = ImageData(input_img, 0)
    all_features.extend(list(np.hstack(img.getORB())))

    if img.img is None:
        print('!ERROR! Image read error!')
        return False

    v_bloc_h = int(img.img_h / 3)
    v_bloc_w = int(img.img_w / 3)
    q_bloc_h = int(v_bloc_h / 2)
    q_bloc_w = int(v_bloc_w / 2)
    q2_bloc_h = int(q_bloc_h / 2)
    q2_bloc_w = int(q_bloc_w / 2)
    q4_bloc_h = int(q2_bloc_h / 2)
    q4_bloc_w = int(q2_bloc_w / 2)

    # check_step = [[h],[w]]
    check_step = np.array([[v_bloc_h, q_bloc_h, q2_bloc_h, q4_bloc_h], [v_bloc_w, q_bloc_w, q2_bloc_w, q4_bloc_w]])

    for i in range(0, check_list.shape[0]):
        j_img = img.img_gray
        for j in range(0, check_list.shape[1]):
            while True:
                if j == 0:
                    this_bloc = p_3_3[check_list[i, j]]
                else:
                    this_bloc = p_2_2[check_list[i, j]]
                # this_bloc = (row index number, col index number) example: (0,2)
                start_h = this_bloc[0] * check_step[0, j]  # row*h
                start_w = this_bloc[1] * check_step[1, j]  # col*w
                end_h = start_h + check_step[0, j]
                end_w = start_w + check_step[1, j]
                this_img = j_img[start_h:end_h, start_w:end_w]
                if debug:
                    cv2.imshow('i=' + str(i) + ',j=' + str(j) + ',bloc' + str(this_bloc), this_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                temp_img = ImageData(this_img, 0)
                temp_density = temp_img.getDensity()
                print('this m cell density:', temp_density)
                if temp_density < check_density:
                    if j == 0:
                        check_list[i, j] = random.randint(0, 9 - 1)
                    else:
                        check_list[i, j] = random.randint(0, 4 - 1)
                else:
                    all_features.extend(list(np.hstack(temp_img.getSURF())))
                    j_img = this_img
                    break

    output_name = S_index + '.csv'
    save_image_features_csv(main_path, output_name, name_index, all_features, result_path='Pyramid')

    return True


def main(args):
    name_index = None
    pyramid_features(args.main_path, args.input_img, name_index, result_path='Pyramid')


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', type=str, nargs='?', default=r'', help='The Main Folder Path')
    parser.add_argument('--input_img', type=str, nargs='?', default=r'', help='The well image SSSS path')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
