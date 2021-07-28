import sys
import os
import time
import argparse
import pandas as pd
from Lib_Class import ImageData
from Lib_Function import image_my_enhancement, image_my_PGC, stitching_well_by_name
from Lib_Features import sigle_features


def main(args):
    # this program can stitching 96 well
    global main_path, well_image
    MyPGC_img = False

    main_path = args.main_path  # r'D:\pro\CD22'
    well_image = []
    well_image.append(args.well_image_0)
    well_image.append(args.well_image_1)

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    if len(well_image) == 1:
        pass
    elif len(well_image) == 2:
        if os.path.exists(well_image[0]):
            t_path_list = os.path.split(well_image[0])  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
            t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
            t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
            SSS_folder = t2_path_list[1]  # 'SSS_100%'
            S_index = t1_path_list[1]  # 'S1'
            S = int(S_index.split('S')[1])  # 1
            img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
            name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'
            T = int(name_index.split('~T')[1])  # 1

            if MyPGC_img:
                path = os.path.join(main_path, 'MyPGC_img', SSS_folder)
            else:
                path = os.path.join(main_path, SSS_folder)
            if S == 96:
                stitching_well_by_name(main_path, path, 'All_Wells', 8, 12, img_name, w=1000, h=1000, zoom=None,
                                       sort_function=None)

        return True

    return False


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', type=str, nargs='?', default=r'', help='The Main Folder Path')
    parser.add_argument('--well_image_0', type=str, nargs='?', default=r'', help='The whole well image SSS')
    parser.add_argument('--well_image_1', type=str, nargs='?', default=r'', help='The whole well square image SSSS')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
