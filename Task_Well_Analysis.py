import sys
import os
import time
import argparse
import pandas as pd
from Lib_Class import ImageData
from Lib_Function import image_my_enhancement, image_my_PGC
from Lib_Features import sigle_features


def main(args):
    # this program can extraction cell density(1) and SIFT features(256) of one M
    global main_path, well_image

    main_path = args.main_path  # r'D:\pro\CD22'
    well_image = []
    well_image.append(args.well_image_0)
    well_image.append(args.well_image_1)
    folder_Enhanced_img = 'Enhanced_img'
    folder_MyPGC_img = 'MyPGC_img'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    if len(well_image) == 1:

        if os.path.exists(well_image[0]):
            t_path_list = os.path.split(well_image[0])  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
            t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
            t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
            SSS_folder = t2_path_list[1]  # 'SSS_100%'
            S_index = t1_path_list[1]  # 'S1'
            img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
            name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

            print('>>>', S_index, '---', name_index, ' Image Enhancement :::')

            # do something :::
            # 1.image_my_enhancement()
            # 2.image_my_PGC()

            img_file = well_image[0]

            to_file = os.path.join(main_path, folder_Enhanced_img, SSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_Enhanced_img)):
                os.makedirs(os.path.join(main_path, folder_Enhanced_img))
            if not os.path.exists(os.path.join(main_path, folder_Enhanced_img, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_Enhanced_img, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_Enhanced_img, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_Enhanced_img, SSS_folder, S_index))
            image_my_enhancement(img_file, to_file)

            to_file = os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index))
            image_my_PGC(img_file, to_file)

    elif len(well_image) == 2:

        if os.path.exists(well_image[0]):
            t_path_list = os.path.split(well_image[0])  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
            t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
            t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
            SSS_folder = t2_path_list[1]  # 'SSS_100%'
            S_index = t1_path_list[1]  # 'S1'
            img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
            name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

            print('>>>', S_index, '---', name_index, ' Image Enhancement :::')

            # do something :::
            # 1.image_my_enhancement()
            # 2.image_my_PGC()

            img_file = well_image[0]

            to_file = os.path.join(main_path, folder_Enhanced_img, SSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_Enhanced_img)):
                os.makedirs(os.path.join(main_path, folder_Enhanced_img))
            if not os.path.exists(os.path.join(main_path, folder_Enhanced_img, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_Enhanced_img, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_Enhanced_img, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_Enhanced_img, SSS_folder, S_index))
            image_my_enhancement(img_file, to_file)

            to_file = os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index))
            image_my_PGC(img_file, to_file)

        if os.path.exists(well_image[1]):
            t_path_list = os.path.split(well_image[1])  # [r'D:\pro\CD22\SSSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
            t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSSS_100%', 'S1']
            t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSSS_100%']
            SSSS_folder = t2_path_list[1]  # 'SSSS_100%'
            S_index = t1_path_list[1]  # 'S1'
            img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
            name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

            print('>>>', S_index, '---', name_index,' Feature Extraction :::')

            # do something :::
            # 0.core_features()
            # 1.image_my_enhancement()
            # 2.core_features()
            # 3.image_my_PGC()
            # 4.core_features()

            img_file = well_image[1]

            sigle_features(main_path, img_file, result_path='Analysis')

            to_file = os.path.join(main_path, folder_Enhanced_img, SSSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_Enhanced_img)):
                os.makedirs(os.path.join(main_path, folder_Enhanced_img))
            if not os.path.exists(os.path.join(main_path, folder_Enhanced_img, SSSS_folder)):
                os.makedirs(os.path.join(main_path, folder_Enhanced_img, SSSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_Enhanced_img, SSSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_Enhanced_img, SSSS_folder, S_index))
            image_my_enhancement(img_file, to_file)
            sigle_features(main_path, to_file, result_path='Enhanced_Analysis')

            to_file = os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index))
            image_my_PGC(img_file, to_file)
            sigle_features(main_path, to_file, result_path='MyPGC_Analysis')

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
