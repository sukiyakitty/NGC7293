import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import cv2
from Lib_Class import ImageData
from Lib_Function import is_number, image_my_enhancement, image_my_PGC, saving_talbe
from Lib_Features import call_matlab_FC


def make_afterCHIR10_CD13_myPGC(main_path, well_image):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False

    # folder_Enhanced_img = 'Enhanced_img'
    folder_MyPGC_img = 'MyPGC_img'
    afterCHIR10_CD13 = ['2018-11-30~IPS-3_CD13~T18', '2018-12-01~I-1_CD13~T1', '2018-12-01~I-1_CD13~T2',
                        '2018-12-01~I-1_CD13~T3', '2018-12-01~I-1_CD13~T4', '2018-12-01~I-1_CD13~T5',
                        '2018-12-01~I-1_CD13~T6', '2018-12-01~I-1_CD13~T7', '2018-12-01~I-1_CD13~T8',
                        '2018-12-01~I-1_CD13~T9', '2018-12-01~I-1_CD13~T10']

    # if len(well_image) == 2:
    #     pass
    # else:
    #     return False

    if os.path.exists(well_image):
        t_path_list = os.path.split(well_image)  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
        t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
        t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
        SSS_folder = t2_path_list[1]  # 'SSS_100%'
        S_index = t1_path_list[1]  # 'S1'
        img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
        name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

        if name_index in afterCHIR10_CD13:
            print('>>>', S_index, '---', name_index, ' Image MyPGC Enhancement :::')
            to_file = os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index))
            image_my_PGC(well_image, to_file)

    return True


def make_afterCHIR10_CD27_myPGC(main_path, well_image):
    # main_path: r'M:\CD27\PROCESSING'
    # well_image: specific image path r'M:\CD27\PROCESSING\SSS_100%\S2\2019-06-22~CD27_stageI_0h~T6.png'
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False

    folder_MyPGC_img = 'MyPGC_img'
    afterCHIR10_CD27 = ['2019-06-22~CD27_IPS(H9)~T1', '2019-06-22~CD27_stageI_0h~T1', '2019-06-22~CD27_stageI_0h~T2',
                        '2019-06-22~CD27_stageI_0h~T3', '2019-06-22~CD27_stageI_0h~T4', '2019-06-22~CD27_stageI_0h~T5',
                        '2019-06-22~CD27_stageI_0h~T6', '2019-06-22~CD27_stageI_0h~T7', '2019-06-22~CD27_stageI_0h~T8',
                        '2019-06-22~CD27_stageI_0h~T9', '2019-06-22~CD27_stageI_0h~T10']

    if os.path.exists(well_image):
        t_path_list = os.path.split(well_image)  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
        t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
        t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
        SSS_folder = t2_path_list[1]  # 'SSS_100%'
        S_index = t1_path_list[1]  # 'S1'
        img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
        name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

        if name_index in afterCHIR10_CD27:
            print('>>>', S_index, '---', name_index, ' Image MyPGC Enhancement :::')
            to_file = os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index))
            image_my_PGC(well_image, to_file)

    return True


def core_features_one_image(main_path, well_image, features=7, result_path='Features'):
    # this program is design for core features extraction; one time one well one time point
    # now it is doing feature extraction
    # input well_image is a str just like r'M:\CD27\PROCESSING\SSSS_100%\S1\2019-06-22~CD27_IPS(H9)~T1.png'
    # features=0b

    result_path = os.path.join(main_path, result_path)
    if os.path.exists(result_path):
        # shutil.rmtree(output_folder)
        pass
    else:
        os.makedirs(result_path)

    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False

    t0_path_list = os.path.split(well_image)  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
    t1_path_list = os.path.split(t0_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
    t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
    Szoom_folder = t2_path_list[1]  # 'SSS_100%'
    Sframe = Szoom_folder.split('_')[0]  # 'SSS'
    S_index = t1_path_list[1]  # 'S1'
    img_name = t0_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
    name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

    Szoom_result_path = os.path.join(result_path, Szoom_folder)
    if os.path.exists(Szoom_result_path):
        # shutil.rmtree(output_folder)
        pass
    else:
        os.makedirs(Szoom_result_path)

    print('>>>core_features_one_image():', Szoom_folder, '===', S_index, '---', name_index)

    this_image = ImageData(well_image)

    # this_image_SIFT = this_image.getSIFT()
    # this_image_SURF = this_image.getSURF()
    # this_image_ORB = this_image.getORB()
    # this_image_density = this_image.getDensity()
    # this_image_perimeter = this_image.getPerimeter()

    if features & 0b1 > 0:

        this_image_SIFT = this_image.getSIFT()
        this_image_SURF = this_image.getSURF()
        this_image_ORB = this_image.getORB()

        Scsv_file = os.path.join(Szoom_result_path, S_index + '.csv')

        if not os.path.exists(Scsv_file):
            Scsv_mem = pd.DataFrame(
                columns=['sift' + str(col) for col in range(1, 1 + this_image_SIFT.shape[0])] +
                        ['sur' + str(col) for col in range(1, 1 + this_image_SURF.shape[0])] +
                        ['orb' + str(col) for col in range(1, 1 + this_image_ORB.shape[0])])
        else:
            Scsv_mem = pd.read_csv(Scsv_file, header=0, index_col=0)

        Scsv_mem.loc[name_index] = np.hstack([this_image_SIFT, this_image_SURF, this_image_ORB])
        Scsv_mem.to_csv(path_or_buf=Scsv_file)

    if features & 0b10 > 0:

        this_image_density = this_image.getDensity()

        if Sframe == 'SSS':
            print('The Whole_Well_Density:', this_image_density)
            saving_talbe(main_path, 'Whole_Well_Density.csv', name_index, S_index, this_image_density)
        elif Sframe == 'SSSS':
            print('The No_Edge_Density:', this_image_density)
            saving_talbe(main_path, 'No_Edge_Density.csv', name_index, S_index, this_image_density)

    if features & 0b100 > 0:

        this_image_perimeter = this_image.getPerimeter()

        if Sframe == 'SSS':
            print('The Whole_Well_Perimeter:', this_image_perimeter)
            saving_talbe(main_path, 'Whole_Well_Perimeter.csv', name_index, S_index, this_image_perimeter)
        elif Sframe == 'SSSS':
            print('The No_Edge_Perimeter:', this_image_perimeter)
            saving_talbe(main_path, 'No_Edge_Perimeter.csv', name_index, S_index, this_image_perimeter)

    return True


def make_afterCHIR_CD39_myPGC(main_path, well_image):
    # main_path: r'D:\CD39\PROCESSING'
    # well_image: specific image path

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False

    folder_MyPGC_img = 'MyPGC_img'
    afterCHIR_CD39 = ['2019-12-12~CD39_IPS_[]~T1', '2019-12-13~CD39_STAGE1_1H_[]~T1', '2019-12-13~CD39_STAGE1_1H_[]~T2',
                      '2019-12-13~CD39_STAGE1_1H_[]~T3', '2019-12-13~CD39_STAGE1_1H_[]~T4',
                      '2019-12-13~CD39_STAGE1_1H_[]~T5', '2019-12-13~CD39_STAGE1_1H_[]~T6',
                      '2019-12-13~CD39_STAGE1_1H_[]~T7', '2019-12-13~CD39_STAGE1_1H_[]~T8',
                      '2019-12-13~CD39_STAGE1_1H_[]~T9', '2019-12-13~CD39_STAGE1_1H_[]~T10',
                      '2019-12-13~CD39_STAGE1_1H_[]~T11', '2019-12-13~CD39_STAGE1_1H_[]~T12',
                      '2019-12-13~CD39_STAGE1_1H_[]~T13', '2019-12-13~CD39_STAGE1_1H_[]~T14',
                      '2019-12-13~CD39_STAGE1_1H_[]~T15', '2019-12-13~CD39_STAGE1_1H_[]~T16',
                      '2019-12-13~CD39_STAGE1_1H_[]~T17', '2019-12-13~CD39_STAGE1_1H_[]~T18',
                      '2019-12-13~CD39_STAGE1_1H_[]~T19', '2019-12-13~CD39_STAGE1_1H_[]~T20',
                      '2019-12-13~CD39_STAGE1_1H_[]~T21', '2019-12-13~CD39_STAGE1_1H_[]~T22',
                      '2019-12-13~CD39_STAGE1_1H_[]~T23', '2019-12-13~CD39_STAGE1_1H_[]~T24',
                      '2019-12-13~CD39_STAGE1_1H_[]~T25']

    if os.path.exists(well_image):
        t_path_list = os.path.split(well_image)  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
        t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
        t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
        SSS_folder = t2_path_list[1]  # 'SSS_100%'
        S_index = t1_path_list[1]  # 'S1'
        img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
        name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

        if name_index in afterCHIR_CD39:
            print('>>>', S_index, '---', name_index, ' Image MyPGC Enhancement :::')
            to_file = os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index))
            image_my_PGC(well_image, to_file)

    return True


def make_afterCHIR_CD41_myPGC(main_path, well_image):
    # main_path: r'H:\CD41\PROCESSING'
    # well_image: specific image path

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False

    folder_MyPGC_img = 'MyPGC_img'
    afterCHIR_CD41 = ['2020-01-16~CD41_IPS~T1', '2020-01-16~CD41_STAGEI_1H~T1', '2020-01-16~CD41_STAGEI_1H~T2',
                      '2020-01-16~CD41_STAGEI_1H~T3', '2020-01-16~CD41_STAGEI_1H~T4',
                      '2020-01-16~CD41_STAGEI_1H~T5', '2020-01-16~CD41_STAGEI_1H~T6',
                      '2020-01-16~CD41_STAGEI_1H~T7', '2020-01-16~CD41_STAGEI_1H~T8',
                      '2020-01-16~CD41_STAGEI_1H~T9', '2020-01-16~CD41_STAGEI_1H~T10',
                      '2020-01-16~CD41_STAGEI_1H~T11', '2020-01-16~CD41_STAGEI_1H~T12',
                      '2020-01-16~CD41_STAGEI_1H~T13', '2020-01-16~CD41_STAGEI_1H~T14',
                      '2020-01-16~CD41_STAGEI_1H~T15', '2020-01-16~CD41_STAGEI_1H~T16',
                      '2020-01-16~CD41_STAGEI_1H~T17', '2020-01-16~CD41_STAGEI_1H~T18',
                      '2020-01-16~CD41_STAGEI_1H~T19', '2020-01-16~CD41_STAGEI_1H~T20',
                      '2020-01-16~CD41_STAGEI_1H~T21', '2020-01-16~CD41_STAGEI_1H~T22',
                      '2020-01-16~CD41_STAGEI_1H~T23', '2020-01-16~CD41_STAGEI_1H~T24',
                      '2020-01-16~CD41_STAGEI_1H~T25']

    if os.path.exists(well_image):
        t_path_list = os.path.split(well_image)  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
        t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
        t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
        SSS_folder = t2_path_list[1]  # 'SSS_100%'
        S_index = t1_path_list[1]  # 'S1'
        img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
        name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

        if name_index in afterCHIR_CD41:
            print('>>>', S_index, '---', name_index, ' Image MyPGC Enhancement :::')
            to_file = os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index))
            image_my_PGC(well_image, to_file)

    return True


def make_afterCHIR_CD11_myPGC(main_path, well_image):
    # main_path: r'D:\CD11'
    # well_image: specific image path

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False

    folder_MyPGC_img = 'MyPGC_img'

    if os.path.exists(well_image):
        t_path_list = os.path.split(well_image)  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
        t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
        t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
        SSS_folder = t2_path_list[1]  # 'SSS_100%'
        S_index = t1_path_list[1]  # 'S1'
        S = int(S_index.split('S')[1])  # 1
        img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
        name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'
        T = int(name_index.split('~T')[1])  # 1
        if name_index.find('IPS') >= 0:
            E = 1
        else:
            E = 2

        print('>>>', S_index, '---', name_index, ' Image MyPGC Enhancement :::')
        to_file = os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index, img_name)
        if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
            os.makedirs(os.path.join(main_path, folder_MyPGC_img))
        if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder)):
            os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder))
        if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index)):
            os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index))

        image_my_PGC(well_image, to_file)
        call_matlab_FC(main_path, to_file, E, T, S, result_path='FractalCurving', this_async=True, do_curving_now=False,
                       sleep_s=0)

    return True


def main(args):
    # args.analysis_function(args.main_path, args.input_img)
    # make_afterCHIR10_CD13_myPGC(args.main_path, args.input_img)
    # make_afterCHIR10_CD27_myPGC(args.main_path, args.input_img)
    # core_features_one_image(args.main_path, args.input_img, features=0b1, result_path='Features')
    # make_afterCHIR_CD39_myPGC(args.main_path, args.input_img)
    make_afterCHIR_CD11_myPGC(args.main_path, args.input_img)
    # make_afterCHIR_CD41_myPGC(args.main_path, args.input_img)


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_function', type=str, nargs='?', default=r'', help='The Main Function')
    parser.add_argument('--main_path', type=str, nargs='?', default=r'', help='The Main Folder Path')
    parser.add_argument('--input_img', type=str, nargs='?', default=r'', help='The well image SSSS path')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
