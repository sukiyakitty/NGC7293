import os
import time
import random
import numpy as np
import pandas as pd
import cv2
import matlab.engine
from Lib_Class import ImageData, ImageName
from Lib_Function import is_number, image_my_enhancement, image_my_PGC, get_cpu_python_process, saving_density, \
    get_img_density, any_to_image, image_to_gray, saving_talbe, stitching_well_by_name
from Lib_Sort import files_sort_CD09, files_sort_CD11, files_sort_CD13, files_sort_CD26, files_sort_CD27, \
    files_sort_CD39, files_sort_CD41, files_sort_univers
import shutil


def call_gabor_analysis(infile, outfile, k, gk, M):
    # call gabor analysis
    os.system("start /min python gabor.py -infile {} -outfile {} -k {} -gk {} -M {}".format(infile, outfile, k, gk, M))
    # os.system("python3 gabor.py -infile {} -outfile {} -k {} -gk {} -M {}".format(infile, outfile, k, gk, M))
    return True


def video_density_flow_test(main_path, input_avi):
    # video
    cap = cv2.VideoCapture(input_avi)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frame = 1
    while True:
        ret, frame2 = cap.read()
        if ret:
            frame = frame + 1

            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # 返回一个两通道的光流向量，实际上是每个点的像素位移值
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 笛卡尔坐标转换为极坐标，获得极轴和极角
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            hsv[..., 0] = ang * 180 / np.pi / 2  # 角度
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # cv2.imshow('frame2', bgr)
            # cv2.imshow("frame1", frame2)
            # print(ret)
            cv2.imwrite(os.path.join(main_path, 'opticalfb_' + str(frame) + '.png'), bgr)

            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
            # elif k == ord('s'):
            #     cv2.imwrite(os.path.join(main_path,'opticalfb.png'), frame2)
            #     cv2.imwrite(os.path.join(main_path,'opticalhsv.png'), bgr)
            prvs = next
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    return True


def get_density_perimeter_save(main_path, well_image):
    #
    # well_image is [r'E:\CD26\Processing\SSS_100%\S9\2019-06-14~CD26_STAGEI_0H~T1.png',
    #                r'E:\CD26\Processing\SSSS_100%\S9\2019-06-14~CD26_STAGEI_0H~T1.png']

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    if not type(well_image) is list:
        print('!ERROR! The well_image is not list!')
        return False

    has_SSS = False
    has_SSSS = False
    if len(well_image) == 1:
        has_SSS = True
    elif len(well_image) == 2:
        has_SSS = True
        has_SSSS = True
    else:
        print('!ERROR! The well_image list error!')
        return False

    row_index = os.path.split(well_image[0])[1].split('.')[0]  # file name without Suffix name
    col_index = os.path.split(os.path.split(well_image[0])[0])[1]  # 'S1'

    if has_SSS:
        img_SSS = ImageData(well_image[0])
        # zws_density = get_img_density(well_image[0])
        SSS_density = img_SSS.getDensity()
        SSS_perimeter = img_SSS.getPerimeter()

        print('The Whole_Well_Density:', SSS_density)
        saving_density(main_path, 'Whole_Well_Density.csv', row_index, col_index, SSS_density)
        print('The Whole_Well_Perimeter:', SSS_perimeter)
        saving_density(main_path, 'Whole_Well_Perimeter.csv', row_index, col_index, SSS_perimeter)

    if has_SSSS:
        img_SSSS = ImageData(well_image[1])
        # znes_density = get_img_density(well_image[1])
        SSSS_density = img_SSSS.getDensity()
        SSSS_perimeter = img_SSSS.getPerimeter()

        print('The No_Edge_Density:', SSSS_density)
        saving_density(main_path, 'No_Edge_Density.csv', row_index, col_index, SSSS_density)
        print('The No_Edge_Perimeter:', SSSS_perimeter)
        saving_density(main_path, 'No_Edge_Perimeter.csv', row_index, col_index, SSSS_perimeter)

        # sigle_features(main_path, well_image[1], result_path='Analysis')

    return True


def get_density_save(main_path, well_image):
    #
    # well_image is [r'E:\CD26\Processing\SSS_100%\S9\2019-06-14~CD26_STAGEI_0H~T1.png',
    #                r'E:\CD26\Processing\SSSS_100%\S9\2019-06-14~CD26_STAGEI_0H~T1.png']

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    if not type(well_image) is list:
        print('!ERROR! The well_image is not list!')
        return False

    has_SSS = False
    has_SSSS = False
    if len(well_image) == 1:
        has_SSS = True
    elif len(well_image) == 2:
        has_SSS = True
        has_SSSS = True
    else:
        print('!ERROR! The well_image list error!')
        return False

    row_index = os.path.split(well_image[0])[1].split('.')[0]
    col_index = os.path.split(os.path.split(well_image[0])[0])[1]

    if has_SSS:
        zws_density = get_img_density(well_image[0])
        print('The Whole_Well_Density:', zws_density)
        saving_density(main_path, 'Whole_Well_Density.csv', row_index, col_index, zws_density)

    if has_SSSS:
        znes_density = get_img_density(well_image[1])
        print('The No_Edge_Density:', znes_density)
        saving_density(main_path, 'No_Edge_Density.csv', row_index, col_index, znes_density)

        # sigle_features(main_path, well_image[1], result_path='Analysis')

    return True


def get_density_and_features_save(main_path, well_image):
    #
    # well_image is [r'E:\CD26\Processing\SSS_100%\S9\2019-06-14~CD26_STAGEI_0H~T1.png',
    #                r'E:\CD26\Processing\SSSS_100%\S9\2019-06-14~CD26_STAGEI_0H~T1.png']

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    if not type(well_image) is list:
        print('!ERROR! The well_image is not list!')
        return False

    has_SSS = False
    has_SSSS = False
    if len(well_image) == 1:
        has_SSS = True
    elif len(well_image) == 2:
        has_SSS = True
        has_SSSS = True
    else:
        print('!ERROR! The well_image list error!')
        return False

    row_index = os.path.split(well_image[0])[1].split('.')[0]
    col_index = os.path.split(os.path.split(well_image[0])[0])[1]

    if has_SSS:
        zws_density = get_img_density(well_image[0])
        print('The Whole_Well_Density:', zws_density)
        saving_density(main_path, 'Whole_Well_Density.csv', row_index, col_index, zws_density)

    if has_SSSS:
        znes_density = get_img_density(well_image[1])
        print('The No_Edge_Density:', znes_density)
        saving_density(main_path, 'No_Edge_Density.csv', row_index, col_index, znes_density)

        sigle_features(main_path, well_image[1], result_path='Analysis')

    return True


def core_analysis(main_path, well_image, result_path='Analysis'):
    # old
    # this program is design for core analysis; one time one well one time point
    # now it is doing feature extraction
    # well_image is a list of path
    # input well_image[0] has well edge, is 5*5
    # input well_image[1] is the 3*3 Square

    result_path = os.path.join(main_path, result_path)
    if os.path.exists(result_path):
        # shutil.rmtree(output_folder)
        pass
    else:
        os.makedirs(result_path)  # make the output folder

    if len(well_image) == 2:
        if os.path.exists(well_image[0]) and os.path.exists(well_image[1]):

            t_path_2list = os.path.split(well_image[0])
            S_index = os.path.split(t_path_2list[0])[1]  # this well name
            name_index = t_path_2list[1][:-4]  # this name (include time point )
            this_SSSS_image = ImageData(well_image[1], 0)

            # -- SIFT features extraction
            this_SSSS_image_features = this_SSSS_image.getSIFT()
            print('>>>Processing:', S_index, '---', name_index)
            Scsv_file = os.path.join(result_path, S_index + '.csv')
            if not os.path.exists(Scsv_file):
                Scsv_mem = pd.DataFrame(
                    columns=['f' + str(col) for col in range(1, 1 + this_SSSS_image_features.shape[0])])
            else:
                Scsv_mem = pd.read_csv(Scsv_file, header=0, index_col=0)
            Scsv_mem.loc[name_index] = this_SSSS_image_features
            Scsv_mem.to_csv(path_or_buf=Scsv_file)
            # -- SIFT features extraction

            return True
        else:
            return False
    else:
        return False


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
    density_threshold = 0.3
    check_density = density_threshold
    try_times = 50

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

    for i in range(0, check_list.shape[0]):  # 4 times: 4 look up
        j_img = img.img_gray
        for j in range(0, check_list.shape[1]):  # 5 times zoom in
            j0_block_density = np.zeros(9)
            j0_times = 0
            j0_dog = True
            while True:  # find the density >= check_density
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
                if temp_density < check_density:  # if density < check_density then radom find another bloc
                    if j == 0:
                        if j0_dog:
                            j0_block_density[check_list[i, j]] = temp_density
                            j0_times = j0_times + 1
                            if j0_times >= try_times:
                                j0_block_density.sort()
                                check_density = j0_block_density[-4] - 0.0001
                                j0_dog = False
                        check_list[i, j] = random.randint(0, 9 - 1)
                    else:
                        check_list[i, j] = random.randint(0, 4 - 1)
                else:  # if density fited, then add features and zoom in image
                    all_features.extend(list(np.hstack(temp_img.getSURF())))
                    j_img = this_img
                    break

    output_name = S_index + '.csv'
    save_image_features_csv(main_path, output_name, name_index, all_features, result_path='Pyramid')

    return True


def sigle_features(main_path, sigle_image, result_path='Features'):
    # this program is design for core analysis; one time one well one time point one image
    # now it is doing feature extraction
    # input sigle_image has well edge, is 5*5

    result_path = os.path.join(main_path, result_path)
    if os.path.exists(result_path):
        # shutil.rmtree(output_folder)
        pass
    else:
        os.makedirs(result_path)  # make the output folder

    if os.path.exists(sigle_image):  # r'D:\pro\CD22\SSSS_100%\S1\2018-11-28~IPS_CD13~T1.jpg'
        t_path_2list = os.path.split(sigle_image)  # [r'D:\pro\CD22\SSSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
        S_index = os.path.split(t_path_2list[0])[1]  # this well name  'S1'
        name_index = t_path_2list[1][:-4]  # this name (include time point ) but no 'S1~'  '2018-11-28~IPS_CD13~T1'
        this_SSSS_image = ImageData(sigle_image, 0)

        this_SSSS_image_SIFT = this_SSSS_image.getSIFT()
        this_SSSS_image_SURF = this_SSSS_image.getSURF()
        this_SSSS_image_ORB = this_SSSS_image.getORB()

        print('>>>Processing:', S_index, '---', name_index)
        Scsv_file = os.path.join(result_path, S_index + '.csv')
        if not os.path.exists(Scsv_file):
            Scsv_mem = pd.DataFrame(
                columns=['sift' + str(col) for col in range(1, 1 + this_SSSS_image_SIFT.shape[0])] +
                        ['sur' + str(col) for col in range(1, 1 + this_SSSS_image_SURF.shape[0])] +
                        ['orb' + str(col) for col in range(1, 1 + this_SSSS_image_ORB.shape[0])])
        else:
            Scsv_mem = pd.read_csv(Scsv_file, header=0, index_col=0)

        Scsv_mem.loc[name_index] = np.hstack([this_SSSS_image_SIFT, this_SSSS_image_SURF, this_SSSS_image_ORB])
        Scsv_mem.to_csv(path_or_buf=Scsv_file)

        return True
    else:
        return False


def core_features_one_image(main_path, well_image, features=0b1110000, result_path='Features'):
    # this program is design for core features extraction; one time one well one time point
    # now it is doing feature extraction
    # input well_image is a str just like r'M:\CD27\PROCESSING\SSSS_100%\S1\2019-06-22~CD27_IPS(H9)~T1.png'
    # features=0b
    # 5bit: SIFT\SURF\ORB;
    # 6bit: well_image Density;
    # 7bit: well_image Perimeter;

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

    if features & 0b10000 > 0:  # 5bit: SIFT\SURF\ORB;

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

    if features & 0b100000 > 0:  # 6bit: well_image Density;

        this_image_density = this_image.getDensity()

        if Sframe == 'SSS':
            print('The Whole_Well_Density:', this_image_density)
            saving_talbe(main_path, 'Whole_Well_Density.csv', name_index, S_index, this_image_density)
        elif Sframe == 'SSSS':
            print('The No_Edge_Density:', this_image_density)
            saving_talbe(main_path, 'No_Edge_Density.csv', name_index, S_index, this_image_density)

    if features & 0b1000000 > 0:  # 7bit: well_image Perimeter;

        this_image_perimeter = this_image.getPerimeter()

        if Sframe == 'SSS':
            print('The Whole_Well_Perimeter:', this_image_perimeter)
            saving_talbe(main_path, 'Whole_Well_Perimeter.csv', name_index, S_index, this_image_perimeter)
        elif Sframe == 'SSSS':
            print('The No_Edge_Perimeter:', this_image_perimeter)
            saving_talbe(main_path, 'No_Edge_Perimeter.csv', name_index, S_index, this_image_perimeter)

    return True


def old_core_features(main_path, well_image, result_path='Features'):
    # this program is design for core features extraction; one time one well one time point
    # now it is doing feature extraction
    # well_image is a list of path
    # input well_image[0] has well edge, (especailly is 5*5)
    # input well_image[1]  (especailly is the 3*3 Square)

    result_path = os.path.join(main_path, result_path)
    if os.path.exists(result_path):
        # shutil.rmtree(output_folder)
        pass
    else:
        os.makedirs(result_path)  # make the output folder

    if len(well_image) == 2:
        if os.path.exists(well_image[1]):  # os.path.exists(well_image[0]) and os.path.exists(well_image[1]):

            t_path_2list = os.path.split(well_image[1])
            S_index = os.path.split(t_path_2list[0])[1]  # this well name
            name_index = t_path_2list[1][:-4]  # this name (include time point ) but no 'S1~'
            this_SSSS_image = ImageData(well_image[1], 0)

            this_SSSS_image_SIFT = this_SSSS_image.getSIFT()
            this_SSSS_image_SURF = this_SSSS_image.getSURF()
            this_SSSS_image_ORB = this_SSSS_image.getORB()

            print('>>>Processing:', S_index, '---', name_index)
            Scsv_file = os.path.join(result_path, S_index + '.csv')
            if not os.path.exists(Scsv_file):
                Scsv_mem = pd.DataFrame(
                    columns=['sift' + str(col) for col in range(1, 1 + this_SSSS_image_SIFT.shape[0])] +
                            ['sur' + str(col) for col in range(1, 1 + this_SSSS_image_SURF.shape[0])] +
                            ['orb' + str(col) for col in range(1, 1 + this_SSSS_image_ORB.shape[0])])
            else:
                Scsv_mem = pd.read_csv(Scsv_file, header=0, index_col=0)

            Scsv_mem.loc[name_index] = np.hstack([this_SSSS_image_SIFT, this_SSSS_image_SURF, this_SSSS_image_ORB])
            Scsv_mem.to_csv(path_or_buf=Scsv_file)

            return True
        else:
            return False
    else:
        return False


def RT_PGC_Features(main_path, well_image, analysis=0b11010000):
    # SSSS is Square Sequential Stitching Scene
    # enhanced(my_enhancement & my_PGC) and save SSS iamges
    # enhanced(my_enhancement & my_PGC) and save SSSS iamges and extract features and save
    # well_image is a list of path
    # input well_image[0] has well edge, (especailly is 5*5)
    # input well_image[1]  (especailly is the 3*3 Square)
    # features = 0b11010000
    # 5bit: SIFT\SURF\ORB;
    # 6bit: well_image Density;
    # 7bit: well_image Perimeter;
    # 8bit: Call Matlab Fractal Curving;

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    folder_MyPGC_img = 'MyPGC_img'

    if len(well_image) == 2:

        if os.path.exists(well_image[0]):  # whole well
            t_path_list = os.path.split(well_image[0])  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
            t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
            t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
            SSS_folder = t2_path_list[1]  # 'SSS_100%'
            S_index = t1_path_list[1]  # 'S1'
            S = int(S_index.split('S')[1])  # 1
            img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
            name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'
            T = int(name_index.split('~T')[1])  # 1

            print('>>>', S_index, '---', name_index, 'SSS Image image_my_PGC() :::')

            # SSS is Sequential Stitching Scene : do something :::
            # 1.image_my_PGC()

            img_file = well_image[0]

            to_file = os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index))

            image_my_PGC(img_file, to_file)

        if os.path.exists(well_image[1]):  # no dish_margin well
            t_path_list = os.path.split(well_image[1])  # [r'D:\pro\CD22\SSSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
            t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSSS_100%', 'S1']
            t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSSS_100%']
            SSSS_folder = t2_path_list[1]  # 'SSSS_100%'
            S_index = t1_path_list[1]  # 'S1'
            S = int(S_index.split('S')[1])  # 1
            img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
            name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'
            T = int(name_index.split('~T')[1])  # 1
            if name_index.find('IPS') >= 0 or name_index.find('ips') >= 0:
                E = 1  # IPS stage
            elif name_index.find('I-1_1H') >= 0 or name_index.find('I_1_1H') >= 0 or name_index.find(
                    'Stage-1_1H') >= 0 or name_index.find('Stage_1_1H') >= 0 or name_index.find(
                'STAGE-1_1H') >= 0 or name_index.find('STAGE_1_1H') >= 0 or name_index.find(
                'STAGE1-0h') >= 0 or name_index.find('STAGE1_0h') >= 0 or name_index.find(
                'STAGE1-1h') >= 0 or name_index.find('STAGE1_1h') >= 0 or name_index.find(
                'STAGEI-0h') >= 0 or name_index.find('STAGEI_0h') >= 0 or name_index.find(
                'STAGEI-1h') >= 0 or name_index.find('STAGEI_1h') >= 0 or name_index.find(
                'Stage1-0h') >= 0 or name_index.find('Stage1_0h') >= 0 or name_index.find(
                'Stage1-1h') >= 0 or name_index.find('Stage1_1h') >= 0 or name_index.find(
                'StageI-0h') >= 0 or name_index.find('StageI_0h') >= 0 or name_index.find(
                'StageI-1h') >= 0 or name_index.find('StageI_1h') >= 0 or name_index.find(
                'stage1-0h') >= 0 or name_index.find('stage1_0h') >= 0 or name_index.find(
                'stage1-1h') >= 0 or name_index.find('stage1_1h') >= 0 or name_index.find(
                'stageI-0h') >= 0 or name_index.find('stageI_0h') >= 0 or name_index.find(
                'stageI-1h') >= 0 or name_index.find('stageI_1h') >= 0 or name_index.find(
                'STAGE1-0H') >= 0 or name_index.find('STAGE1_0H') >= 0 or name_index.find(
                'STAGE1-1H') >= 0 or name_index.find('STAGE1_1H') >= 0 or name_index.find(
                'STAGEI-0H') >= 0 or name_index.find('STAGEI_0H') >= 0 or name_index.find(
                'STAGEI-1H') >= 0 or name_index.find('STAGEI_1H') >= 0 or name_index.find(
                'Stage1-0H') >= 0 or name_index.find('Stage1_0H') >= 0 or name_index.find(
                'Stage1-1H') >= 0 or name_index.find('Stage1_1H') >= 0 or name_index.find(
                'StageI-0H') >= 0 or name_index.find('StageI_0H') >= 0 or name_index.find(
                'StageI-1H') >= 0 or name_index.find('StageI_1H') >= 0 or name_index.find(
                'stage1-0H') >= 0 or name_index.find('stage1_0H') >= 0 or name_index.find(
                'stage1-1H') >= 0 or name_index.find('stage1_1H') >= 0 or name_index.find(
                'stageI-0H') >= 0 or name_index.find('stageI_0H') >= 0 or name_index.find(
                'stageI-1H') >= 0 or name_index.find('stageI_1H') >= 0 or name_index.find(
                'Stage-1_1H') >= 0 or name_index.find('Stage-1_0H') >= 0:
                E = 2  # stage1 from 0h to 24h~
            else:
                E = 3  # other stage

            print('>>>', S_index, '---', name_index, 'SSSS Image Feature_Extraction() :::')

            # SSSS is Square Sequential Stitching Scene : do something :::
            # 1.core_features_one_image()
            # 2.image_my_PGC()

            img_file = well_image[1]

            # sigle_features(main_path, img_file, result_path='Analysis')
            core_features_one_image(main_path, img_file, features=analysis & 0b01110000, result_path='Features')

            to_file = os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index))

            image_my_PGC(img_file, to_file)

            if analysis & 0b10000000 != 0 and (E == 1 or E == 2) and T <= 15:  # 4bit: Call Matlab Fractal Curving;
                call_matlab_FC(main_path, to_file, E, T, S, result_path='FractalCurving', this_async=True,
                               do_curving_now=False, sleep_s=0)
            # sigle_features(main_path, to_file, result_path='MyPGC_Analysis')

        return True

    return False


def after_PGC_do_Fractal(main_path, times=15, sort_function=files_sort_univers, sleep_s=1, my_title=''):
    # 1. make a Fractal.csv file and loop fill all
    # 2. draw curves
    # do last iPS and first  times =15 images in stage-1
    # as default:  find folder:  main_path/MyPGC_img/SSSS_100%

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    MyPGC_path = os.path.join(main_path, 'MyPGC_img')
    if not os.path.exists(MyPGC_path):
        print('!ERROR! The MyPGC_path does not existed!')
        return False
    MyPGC_SSSS_path = os.path.join(MyPGC_path, 'SSSS_100%')
    if not os.path.exists(MyPGC_SSSS_path):
        print('!ERROR! The MyPGC_SSSS_path does not existed!')
        return False

    SSSS_path_list = os.listdir(MyPGC_SSSS_path)
    SSSS_path_list.sort(key=lambda x: int(x.split('S')[1]))
    if len(SSSS_path_list) <= 0:
        return False

    # find the Experiment_Plan.csv file
    experiment_file_path = os.path.join(main_path, 'Experiment_Plan.csv')
    if os.path.exists(experiment_file_path):  # existed!
        experiment_plan_pd = pd.read_csv(experiment_file_path, header=0, index_col=0)
        # ['medium', 'IPS_type', 'density', 'truly', 'chir', 'chir_hour', 'rest_hour', 'IF_intensity', 'IF_human']
    else:
        print('!ERROR! The Experiment_Plan.csv does not existed!')
        return False

    # ---the Fractal.csv IS existed? it's structures? ---
    fractal_file_path = os.path.join(main_path, 'Fractal.csv')
    if os.path.exists(fractal_file_path):  # existed!
        fractal_df = pd.read_csv(fractal_file_path, header=0, index_col=0)
        fractal_df = fractal_df.fillna(0)
        fractal_df = fractal_df.applymap(lambda x: float(x))
    else:  # not existed!
        print('The First time to processing Fractal!')

        # extract name list : last iPS and first  times =15 images in stage-1
        first_S = os.path.join(MyPGC_SSSS_path, SSSS_path_list[0])
        file_list = []
        file_ips = None
        find_ips = False

        img_files_list = os.listdir(first_S)
        if sort_function is not None:
            sort_function(img_files_list)
        for this_img_name in img_files_list:
            img_path = os.path.join(first_S, this_img_name)
            pth_name = ImageName(img_path)
            if not find_ips:
                if pth_name.stage == 0:
                    file_ips = pth_name.img_name
                else:
                    file_list.append(file_ips)
                    find_ips = True
            if pth_name.stage == 1 and pth_name.hour <= times and pth_name.T <= times:
                file_list.append(this_img_name)

        # initialize colums of Fractal.csv
        # fractal_df = pd.DataFrame(columns=['iPS'] + ['T' + str(col) for col in range(1, times)])
        fractal_df = pd.DataFrame(columns=file_list)

        # initialize empty Fractal.csv (-1)
        # for each_img_name in file_list:
        #     fractal_df.loc[each_img_name] = -1*np.ones(times+1)
        for each_index in experiment_plan_pd.index:
            fractal_df.loc[each_index] = -1 * np.ones(len(file_list))

        fractal_df.to_csv(path_or_buf=fractal_file_path)  # save

    # fill Fractal.csv
    for each_S in fractal_df.index:
        for each_img in fractal_df.columns:
            each_value = fractal_df.loc[each_S, each_img]
            if each_value <= 0:  # fill
                # print(each_value)
                print('>>>', each_S, '---', each_img, 'after_PGC_do_Fractal() :::')
                this_img_path = os.path.join(MyPGC_SSSS_path, each_S, each_img)
                matlab_engine = matlab.engine.start_matlab()
                matlab_call_return = matlab_engine.Task_Fractal_S(this_img_path)
                # print(matlab_call_return)
                fractal_df.loc[each_S, each_img] = matlab_call_return
                matlab_engine.quit()
                fractal_df.to_csv(path_or_buf=fractal_file_path)  # save
                print('::: Finished! |||')
                time.sleep(sleep_s)

    # all had done draw .csv:[fractal_file_path] mem:[fractal_df]
    draw_path = os.path.join(main_path, 'Fractal_Draw')
    if os.path.exists(draw_path):
        shutil.rmtree(draw_path)
    os.makedirs(draw_path)
    print('>>> Start Drawing :::')
    matlab_engine = matlab.engine.start_matlab()
    matlab_call_return = matlab_engine.Task_Draw(matlab.double(fractal_df.values.tolist()), main_path, my_title, draw_path)
    print(matlab_call_return)
    matlab_engine.quit()
    print('::: Draw Finished! |||')
    time.sleep(sleep_s)

    return True


def renew_one_S_Fractal(main_path, S, image_name):
    # only renew one S Fractal

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    MyPGC_path = os.path.join(main_path, 'MyPGC_img')
    if not os.path.exists(MyPGC_path):
        print('!ERROR! The MyPGC_path does not existed!')
        return False
    MyPGC_SSSS_path = os.path.join(MyPGC_path, 'SSSS_100%')
    if not os.path.exists(MyPGC_SSSS_path):
        print('!ERROR! The MyPGC_SSSS_path does not existed!')
        return False

    # ---the Fractal.csv IS existed? it's structures? ---
    fractal_file_path = os.path.join(main_path, 'Fractal.csv')
    if os.path.exists(fractal_file_path):  # existed!
        fractal_df = pd.read_csv(fractal_file_path, header=0, index_col=0)
        fractal_df = fractal_df.fillna(0)
        fractal_df = fractal_df.applymap(lambda x: float(x))
    else:  # not existed!
        print('!ERROR! The Fractal.csv does not existed!')
        return False

    if type(image_name) is str:
        pass
    else:
        print('!ERROR! The input image_name must be str type!')
        return False

    if type(S) is str:
        if S.find('S') >= 0:
            S_str = S
        else:
            print('!ERROR! The input S_str must have "S"!')
            return False
    elif type(S) is int:
        S_str = 'S' + str(S)
    else:
        print('!ERROR! The input S must be str or int type!')
        return False

    # fill Fractal.csv
    print('>>>', S_str, '---', image_name, 'renew_one_S_Fractal() :::')
    this_img_path = os.path.join(MyPGC_SSSS_path, S_str, image_name)
    matlab_engine = matlab.engine.start_matlab()
    matlab_call_return = matlab_engine.Task_Fractal_S(this_img_path)
    # print(matlab_call_return)
    fractal_df.loc[S_str, image_name] = matlab_call_return
    matlab_engine.quit()
    fractal_df.to_csv(path_or_buf=fractal_file_path)  # save
    print('::: Finished! |||')

    return True


def image_file_name_resolving(img_path):
    # temp code NO USING!
    result = None
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    t_path_list = os.path.split(img_path)  # [r'D:\pro\CD22\SSSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
    t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSSS_100%', 'S1']
    t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSSS_100%']
    SSSS_folder = t2_path_list[1]  # 'SSSS_100%'
    S_index = t1_path_list[1]  # 'S1'
    S = int(S_index.split('S')[1])  # 1
    img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
    name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'
    T = int(name_index.split('~T')[1])  # 1

    return result


def core_features_enhanced(main_path, well_image):
    # enhanced(my_enhancement & my_PGC) and save SSS iamges
    # enhanced(my_enhancement & my_PGC) and save SSSS iamges and extract features and save
    # well_image is a list of path
    # input well_image[0] has well edge, (especailly is 5*5)
    # input well_image[1]  (especailly is the 3*3 Square)

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    folder_Enhanced_img = 'Enhanced_img'
    folder_MyPGC_img = 'MyPGC_img'

    if len(well_image) == 2:

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

            print('>>>', S_index, '---', name_index, ' Feature Extraction :::')

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
            # core_features(main_path, well_image, result_path='Analysis')
            sigle_features(main_path, to_file, result_path='Enhanced_Analysis')

            to_file = os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index))
            image_my_PGC(img_file, to_file)
            # core_features(main_path, well_image, result_path='Analysis')
            sigle_features(main_path, to_file, result_path='MyPGC_Analysis')

        return True

    return False


def ext_images_features_high_contrast(main_path, well_image):
    # enhanced(high_contrast) and save SSS iamges
    # enhanced(high_contrast) and save SSSS iamges and extract features and save
    # well_image is a list of path
    # input well_image[0] has well edge, (especailly is 5*5)
    # input well_image[1]  (especailly is the 3*3 Square)

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    folder_MyPGC_img = 'MyPGC_img'

    if len(well_image) == 1:

        if os.path.exists(well_image[0]):
            t_path_list = os.path.split(well_image[0])  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
            t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
            t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
            SSS_folder = t2_path_list[1]  # 'SSS_100%'
            S_index = t1_path_list[1]  # 'S1'
            img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
            name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

            print('>>>', S_index, '---', name_index, ' Image high contrast :::')

            # do something :::
            # 1.image_my_PGC()

            img_file = well_image[0]

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

            print('>>>', S_index, '---', name_index, ' Image high contrast :::')

            # do something :::
            # 1.image_my_PGC()

            img_file = well_image[0]

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

            print('>>>', S_index, '---', name_index, ' Feature Extraction :::')

            # do something :::
            # 0.core_features()
            # 1.image_my_PGC()
            # 2.core_features()

            img_file = well_image[1]

            sigle_features(main_path, img_file, result_path='Analysis')

            to_file = os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index, img_name)
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSSS_folder)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSSS_folder, S_index))
            image_my_PGC(img_file, to_file)
            # core_features(main_path, well_image, result_path='Analysis')
            sigle_features(main_path, to_file, result_path='MyPGC_Analysis')

        return True

    return False


def merge_specific_time_point_features(main_path, features_path, sp_tp, output_name):
    # merge specific time point features of all well(S)
    # input:
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # features_path:  r'C:\Users\Kitty\Desktop\CD13\Analysis'
    # output_name: 'specific_FEATURES.csv'
    # sp_tp: ['2018-11-30~IPS-3_CD13~T13',...]
    # output:
    # (main_path, output_name)
    # output_name 'output.csv'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(features_path):
        print('!ERROR! The features_path does not existed!')
        return False
    if not isinstance(sp_tp, list):
        sp_tp = [sp_tp]
    if not isinstance(sp_tp[0], str):
        print('!ERROR! Input specific time point must be str, EXP: [2018-11-30~IPS-3_CD13~T13] !')
        return False

    result_data = None

    features_CSVs = os.listdir(features_path)
    features_CSVs.sort(key=lambda x: int(x.split('.csv')[0].split('S')[1]))  # ['S1.csv','S2.csv','S3.csv'...]
    for features_csv in features_CSVs:  # 'S1.csv'
        this_features_DF = pd.read_csv(os.path.join(features_path, features_csv), header=0, index_col=0)
        this_features_DF = this_features_DF.applymap(is_number)
        this_features_DF = this_features_DF.dropna(axis=0, how='any')

        findable_sp_tp = []
        for i in sp_tp:  # ['2018-11-30~IPS-3_CD13~T13','2018-11-30~IPS-3_CD13~T14'...]
            if i in this_features_DF.index:
                findable_sp_tp.append(i)
            else:
                i_list = i.split('~T')
                new_i = '~T'.join([i_list[0], str(int(i_list[1]) - 1)])
                print('!Notice! ', features_csv, 'did not contain time point:', i, '! Using', new_i, 'instead!')
                if new_i not in this_features_DF.index:
                    print('!ERROR! Please check your time point name! It was not found!')
                    return False
                findable_sp_tp.append(new_i)

        dst_index = [features_csv.split('.csv')[0] + '~' + i for i in findable_sp_tp]
        this_rows_data = pd.DataFrame(this_features_DF.loc[findable_sp_tp, :])
        if len(this_rows_data) != len(findable_sp_tp):
            this_rows_data = this_rows_data.T
        this_rows_data.index = dst_index

        if result_data is None:
            result_data = this_rows_data
        else:
            result_data = result_data.append(this_rows_data)

    result_data.to_csv(path_or_buf=os.path.join(main_path, output_name))

    return True


def merge_all_well_features(main_path, features_path, output_name='All_FEATURES.csv'):
    # merge all well(S) features, all time point
    # input:
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # features_path:  r'C:\Users\Kitty\Desktop\CD13\Analysis'
    # output_name: 'All_DATA.csv'
    # output:
    # (main_path, output_name)
    # output_name: 'All_FEATURES.csv'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    features_path = os.path.join(main_path, features_path)
    if not os.path.exists(features_path):
        print('!ERROR! The features_path does not existed!')
        return False

    all_data = None

    analysis_CSVs = os.listdir(features_path)
    analysis_CSVs.sort(key=lambda x: int(x.split('.csv')[0].split('S')[1]))
    for ana_csv in analysis_CSVs:
        this_features = pd.read_csv(os.path.join(features_path, ana_csv), header=0, index_col=0)
        this_features = this_features.applymap(is_number)
        this_features = this_features.dropna(axis=0, how='any')
        my_index = this_features.index
        my_index = ana_csv[:-4] + '~' + my_index
        this_features.index = my_index

        if all_data is None:
            all_data = this_features
        else:
            all_data = all_data.append(this_features)
    all_data.to_csv(path_or_buf=os.path.join(main_path, output_name))

    return True


def research_stitched_image_elastic_bat(main_path, zoom, sort_function, analysis_function, do_SSS=True, do_SSSS=True,
                                        do_parallel=False, process_number=12):
    # core methed!
    # this program is design for go through all the SSS and SSSS images, and do core_features()
    # always after experiment, using the whole SSSS images
    # once one image
    # input main_path is the main path
    # input zoom is the zoom
    # output is True or False

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    zoom_str = "%.0f%%" % (zoom * 100)
    SSS_path = os.path.join(main_path, 'SSS_' + zoom_str)
    SSSS_path = os.path.join(main_path, 'SSSS_' + zoom_str)
    has_SSS = do_SSS
    has_SSSS = do_SSSS

    if not os.path.exists(SSS_path):
        print('!Caution! The', SSS_path, 'does not existed!')
        has_SSS = False
    if not os.path.exists(SSSS_path):
        print('!Caution! The', SSSS_path, 'does not existed!')
        has_SSSS = False

    if has_SSS:
        path_list = os.listdir(SSS_path)
        path_list.sort(key=lambda x: int(x.split('S')[1]))
        for this_S_folder in path_list:  # S1 to S96
            Spath = os.path.join(SSS_path, this_S_folder)
            img_files_list = os.listdir(Spath)
            sort_function(img_files_list)
            for img in img_files_list:  # all time sequence
                input_img = os.path.join(Spath, img)
                if do_parallel:
                    call_do_parallel(analysis_function, main_path, input_img)
                    multithreading_control(process_number=process_number)
                else:
                    analysis_function(main_path, input_img)

    if has_SSSS:
        path_list = os.listdir(SSSS_path)
        path_list.sort(key=lambda x: int(x.split('S')[1]))
        for this_S_folder in path_list:  # S1 to S96
            Spath = os.path.join(SSSS_path, this_S_folder)
            img_files_list = os.listdir(Spath)
            sort_function(img_files_list)
            for img in img_files_list:  # all time sequence
                input_img = os.path.join(Spath, img)
                if do_parallel:
                    call_do_parallel(analysis_function, main_path, input_img)
                    multithreading_control(process_number=process_number)
                else:
                    analysis_function(main_path, input_img)

    return True


def multithreading_control(process_number=12):
    temp_p_number = get_cpu_python_process()
    print('!Notice! : $$$---Python Number: ', temp_p_number, '---$$$')
    if temp_p_number > process_number:
        print('!Notice! : Python process number is more than ', process_number, ' Now sleep!')
        time.sleep(5)
        while True:
            if get_cpu_python_process() >= process_number:
                time.sleep(5)
            else:
                break
    return True


def call_do_parallel(analysis_function, main_path, input_img):
    # call parallel
    os.system("start /min python Task_Do_Parallel.py --main_path {} --input_img {}".format(main_path, input_img))


def call_pyramid_features(main_path, input_img):
    # call able analysis
    os.system("start /min python Task_Pyramid_Features.py --main_path {} --input_img {}".format(main_path, input_img))


def old_research_stitched_image(main_path, zoom, sort_function, core_function):
    # core methed!
    # this program is design for go through all the SSS and SSSS images, and do core_features()
    # always after experiment, using the whole SSSS images
    # once one image
    # input main_path is the main path
    # input zoom is the zoom
    # output is True or False

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    zoom_str = "%.0f%%" % (zoom * 100)
    SSS_path = os.path.join(main_path, 'SSS_' + zoom_str)
    SSSS_path = os.path.join(main_path, 'SSSS_' + zoom_str)
    has_SSS = True
    has_SSSS = True

    if not os.path.exists(SSS_path):
        print('!Caution! The', SSS_path, 'does not existed!')
        has_SSS = False
    if not os.path.exists(SSSS_path):
        print('!Caution! The', SSSS_path, 'does not existed!')
        has_SSSS = False
        # return False

    if has_SSS:
        SSS_path_list = os.listdir(SSS_path)
        SSS_path_list.sort(key=lambda x: int(x.split('S')[1]))
        for this_S_folder in SSS_path_list:  # S1 to S96
            Spath = os.path.join(SSS_path, this_S_folder)
            img_files_list = os.listdir(Spath)
            # CD13 style
            # '2018-11-28~IPS_CD13~T2.jpg' '2018-11-28~IPS-2_CD13~T13.jpg'
            sort_function(img_files_list)
            # img_files_list.sort(
            #     key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
            #                    int(x.split('~')[0].split('-')[2]),
            #                    int(x.split('~')[1].split('_')[0].split('-')[1]) if (
            #                            x.split('~')[1].split('_')[0].find('-') >= 0) else 0,
            #                    int(x.split('~T')[1].split('.')[0])])
            for img in img_files_list:  # all the time point images
                well_image = []
                well_image.append(os.path.join(SSS_path, this_S_folder, img))
                if has_SSSS:
                    well_image.append(os.path.join(SSSS_path, this_S_folder, img))

                # core_analysis(main_path, well_image)
                core_function(main_path, well_image)

    return True


def old_research_stitched_SSSS_image(main_path, zoom):
    # old methed!
    # this program is design for go through all the SSS and SSSS images, and do core_features()
    # always after experiment, using the whole SSSS images
    # once one image
    # input main_path is the main path
    # input zoom is the zoom
    # output is True or False

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    zoom_str = "%.0f%%" % (zoom * 100)
    # SSS_path = os.path.join(main_path, 'SSS_' + zoom_str)
    SSSS_path = os.path.join(main_path, 'SSSS_' + zoom_str)

    # if not os.path.exists(SSS_path):
    #     print('!ERROR! The', SSS_path, 'does not existed!')
    #     return False
    if not os.path.exists(SSSS_path):
        print('!ERROR! The', SSSS_path, 'does not existed!')
        return False

    SSSS_path_list = os.listdir(SSSS_path)
    SSSS_path_list.sort(key=lambda x: int(x.split('S')[1]))
    for this_S_folder in SSSS_path_list:  # S1 to S96
        Spath = os.path.join(SSSS_path, this_S_folder)
        img_files_list = os.listdir(Spath)
        # CD13 style
        # '2018-11-28~IPS_CD13~T2.jpg' '2018-11-28~IPS-2_CD13~T13.jpg'
        img_files_list.sort(
            key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                           int(x.split('~')[0].split('-')[2]),
                           int(x.split('~')[1].split('_')[0].split('-')[1]) if (
                                   x.split('~')[1].split('_')[0].find('-') >= 0) else 0,
                           int(x.split('~T')[1].split('.')[0])])
        for img in img_files_list:  # all the time point images
            print('>>>Processing:', this_S_folder, '---', img)
            pyramid_features(main_path, os.path.join(SSSS_path, this_S_folder, img), None, result_path='Pyramid')

    return True


def old_research_stitched_SSSS_image_parallel(main_path, zoom):
    # old methed!
    # this program is design for go through all the SSS and SSSS images, and do core_features()
    # always after experiment, using the whole SSSS images
    # once one image
    # input main_path is the main path
    # input zoom is the zoom
    # output is True or False

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    zoom_str = "%.0f%%" % (zoom * 100)
    # SSS_path = os.path.join(main_path, 'SSS_' + zoom_str)
    SSSS_path = os.path.join(main_path, 'SSSS_' + zoom_str)

    # if not os.path.exists(SSS_path):
    #     print('!ERROR! The', SSS_path, 'does not existed!')
    #     return False
    if not os.path.exists(SSSS_path):
        print('!ERROR! The', SSSS_path, 'does not existed!')
        return False

    SSSS_path_list = os.listdir(SSSS_path)
    SSSS_path_list.sort(key=lambda x: int(x.split('S')[1]))
    for this_S_folder in SSSS_path_list:  # S1 to S96
        Spath = os.path.join(SSSS_path, this_S_folder)
        img_files_list = os.listdir(Spath)
        # CD13 style
        # '2018-11-28~IPS_CD13~T2.jpg' '2018-11-28~IPS-2_CD13~T13.jpg'
        img_files_list.sort(
            key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                           int(x.split('~')[0].split('-')[2]),
                           int(x.split('~')[1].split('_')[0].split('-')[1]) if (
                                   x.split('~')[1].split('_')[0].find('-') >= 0) else 0,
                           int(x.split('~T')[1].split('.')[0])])
        for img in img_files_list:  # all the time point images
            print('>>>Processing:', this_S_folder, '---', img)
            call_pyramid_features(main_path, os.path.join(SSSS_path, this_S_folder, img))
            # pyramid_features(main_path, os.path.join(SSSS_path, this_S_folder, img), None, result_path='Pyramid')
            multithreading_control(process_number=12)

    return True


def old_research_stitched_image_enhanced(main_path, zoom):
    # old methed!
    # this program is design for go through all the SSS and SSSS images, and do core_features_enhanced()
    # always after experiment, using the whole SSSS images
    # once one image
    # input main_path is the main path
    # input zoom is the zoom
    # output is True or False

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    zoom_str = "%.0f%%" % (zoom * 100)
    SSS_path = os.path.join(main_path, 'SSS_' + zoom_str)
    SSSS_path = os.path.join(main_path, 'SSSS_' + zoom_str)

    if not os.path.exists(SSS_path):
        print('!ERROR! The', SSS_path, 'does not existed!')
        return False
    if not os.path.exists(SSSS_path):
        print('!ERROR! The', SSSS_path, 'does not existed!')
        return False

    SSSS_path_list = os.listdir(SSSS_path)
    SSSS_path_list.sort(key=lambda x: int(x.split('S')[1]))
    for this_S_folder in SSSS_path_list:  # S1 to S96
        Spath = os.path.join(SSSS_path, this_S_folder)
        img_files_list = os.listdir(Spath)
        # CD23 style
        # 2019-05-24~CD23_A_72h~T1.png
        # 2019-05-24~CD23_A_D4end~T1.png
        # 2019-05-24~CD23_A_D5end~T1.png
        img_files_list.sort(
            key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                           int(x.split('~')[0].split('-')[2]), int(x.split('~T')[1].split('.')[0])])
        for img in img_files_list:  # all the time point images
            well_image = []
            well_image.append(os.path.join(SSS_path, this_S_folder, img))
            well_image.append(os.path.join(SSSS_path, this_S_folder, img))
            core_features_enhanced(main_path, well_image)

    return True


def make_firstIPS_CD13_myPGC(main_path, well_image):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False

    # folder_Enhanced_img = 'Enhanced_img'
    folder_MyPGC_img = 'MyPGC_img'
    afterCHIR10_CD13 = ['2018-11-30~IPS-3_CD13~T18', '2018-12-01~I-1_CD13~T1']

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
        if name_index.find('IPS') >= 0 or name_index.find('ips') >= 0:
            E = 1
        elif name_index.find('I-1') >= 0 or name_index.find('I_1') >= 0 or name_index.find(
                'Stage-1') >= 0 or name_index.find('Stage_1') >= 0 or name_index.find(
            'STAGE-1') >= 0 or name_index.find('STAGE_1') >= 0 or name_index.find(
            'STAGE1-0h') >= 0 or name_index.find('STAGE1_0h') >= 0 or name_index.find(
            'STAGE1-1h') >= 0 or name_index.find('STAGE1_1h') >= 0 or name_index.find(
            'STAGEI-0h') >= 0 or name_index.find('STAGEI_0h') >= 0 or name_index.find(
            'STAGEI-1h') >= 0 or name_index.find('STAGEI_1h') >= 0 or name_index.find(
            'Stage1-0h') >= 0 or name_index.find('Stage1_0h') >= 0 or name_index.find(
            'Stage1-1h') >= 0 or name_index.find('Stage1_1h') >= 0 or name_index.find(
            'StageI-0h') >= 0 or name_index.find('StageI_0h') >= 0 or name_index.find(
            'StageI-1h') >= 0 or name_index.find('StageI_1h') >= 0 or name_index.find(
            'stage1-0h') >= 0 or name_index.find('stage1_0h') >= 0 or name_index.find(
            'stage1-1h') >= 0 or name_index.find('stage1_1h') >= 0 or name_index.find(
            'stageI-0h') >= 0 or name_index.find('stageI_0h') >= 0 or name_index.find(
            'stageI-1h') >= 0 or name_index.find('stageI_1h') >= 0:
            E = 2
        else:
            E = 3

        print('>>>', S_index, '---', name_index, ' Image MyPGC Enhancement :::')
        to_file = os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index, img_name)
        if not os.path.exists(os.path.join(main_path, folder_MyPGC_img)):
            os.makedirs(os.path.join(main_path, folder_MyPGC_img))
        if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder)):
            os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder))
        if not os.path.exists(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index)):
            os.makedirs(os.path.join(main_path, folder_MyPGC_img, SSS_folder, S_index))

        image_my_PGC(well_image, to_file)
        # call_matlab_FC(main_path, to_file, E, T, S, result_path='FractalCurving', this_async=True, do_curving_now=False,
        #                sleep_s=0)

    return True


def make_afterCHIR_reseach_CD09(main_path, well_image):
    # main_path: r'C:\C137\PROCESSING\CD09'
    # well_image: specific image path: r'C:\C137\PROCESSING\CD09\MyPGC_img\SSS_50%\S1\2018-09-02~IPS_CD09~T1.png'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False

    folder_temp = 'folder_temp'

    afterCHIR_CD09 = ['2018-09-02~IPS_CD09~T1', '2018-09-03~I-1_CD09~T1', '2018-09-03~I-1_CD09~T2',
                      '2018-09-03~I-1_CD09~T3', '2018-09-03~I-1_CD09~T4', '2018-09-03~I-1_CD09~T5',
                      '2018-09-03~I-1_CD09~T6', '2018-09-03~I-1_CD09~T7', '2018-09-03~I-1_CD09~T8',
                      '2018-09-03~I-1_CD09~T9', '2018-09-03~I-1_CD09~T10', '2018-09-03~I-1_CD09~T11',
                      '2018-09-03~I-1_CD09~T12', '2018-09-03~I-1_CD09~T13']

    b_list_m = [[0, 1868, 2769, 4611], [0, 3687, 2769, 6447], [0, 5526, 2769, 8268],
                [966, 966, 3682, 3687], [966, 6447, 3682, 9186],
                [1844, 0, 4607, 2765], [1844, 1868, 4607, 4611], [1844, 3687, 4607, 6447], [1844, 5526, 4607, 8268],
                [1844, 7376, 4607, 10142],
                [3682, 0, 6452, 2765], [3682, 1868, 6452, 4611], [3682, 3687, 6452, 6447], [3682, 5526, 6452, 8268],
                [3682, 7376, 6452, 10142],
                [5528, 0, 8265, 2765], [5528, 1868, 8265, 4611], [5528, 3687, 8265, 6447], [5528, 5526, 8265, 8268],
                [5528, 7376, 8265, 10142],
                [6452, 966, 9189, 3687], [6452, 6447, 9189, 9186],
                [7376, 1868, 10142, 4611], [7376, 3687, 10142, 6447], [7376, 5526, 10142, 8268]]

    b_list_e = [[0, 0, 2788, 2766], [0, 1844, 2788, 4612], [0, 3688, 2788, 6429],
                [922, 0, 3729, 2766], [922, 4612, 3729, 7350],
                [1845, 0, 4607, 2766], [1845, 1844, 4607, 4612], [1845, 3688, 4607, 6429], [1845, 5534, 4607, 8298],
                [1845, 2766, 4607, 5534],
                [3729, 0, 6453, 2766], [3729, 1844, 6453, 4612], [3729, 3688, 6453, 6429], [3729, 5534, 6453, 8298],
                [3729, 2766, 6453, 5534],
                [5528, 0, 8272, 2766], [5528, 1844, 8272, 4612], [5528, 3688, 8272, 6429], [5528, 5534, 8272, 8298],
                [5528, 2766, 8272, 5534],
                [6452, 0, 9191, 2766], [6452, 4612, 9191, 7350],
                [7374, 0, 10142, 2766], [7374, 1844, 10142, 4612], [7374, 3688, 10142, 6429]]
    edge = [1, 12, 13, 24]

    if os.path.exists(well_image):  # r'C:\C137\PROCESSING\CD09\MyPGC_img\SSS_50%\S1\2018-09-02~IPS_CD09~T1.png'
        t_path_list = os.path.split(
            well_image)  # [r'C:\C137\PROCESSING\CD09\MyPGC_img\SSS_50%\S1', '2018-09-02~IPS_CD09~T1.png']
        t1_path_list = os.path.split(t_path_list[0])  # [r'C:\C137\PROCESSING\CD09\MyPGC_img\SSS_50%', 'S1']
        t2_path_list = os.path.split(t1_path_list[0])  # [r'C:\C137\PROCESSING\CD09\MyPGC_img', 'SSS_50%']
        SSS_folder = t2_path_list[1]  # 'SSS_50%'
        S_index = t1_path_list[1]  # 'S1'
        S = int(S_index.split('S')[1])  # 1
        img_name = t_path_list[1]  # '2018-09-02~IPS_CD09~T1.png'
        name_index = img_name[:-4]  # '2018-09-02~IPS_CD09~T1'
        # T = int(name_index.split('~T')[1])  # 1
        # '2018-09-17~Result_CD09~T1~C1'
        if (name_index.find('~C') > 0):
            T = int(name_index.split('~T')[1].split('~C')[0])
        else:
            T = int(name_index.split('~T')[1])

        if name_index.find('IPS') >= 0 or name_index.find('ips') >= 0:
            E = 1
        elif name_index.find('I-1') >= 0 or name_index.find('I_1') >= 0 or name_index.find(
                'Stage-1') >= 0 or name_index.find('Stage_1') >= 0 or name_index.find(
            'STAGE-1') >= 0 or name_index.find('STAGE_1') >= 0 or name_index.find(
            'STAGE1-0h') >= 0 or name_index.find('STAGE1_0h') >= 0 or name_index.find(
            'STAGE1-1h') >= 0 or name_index.find('STAGE1_1h') >= 0 or name_index.find(
            'STAGEI-0h') >= 0 or name_index.find('STAGEI_0h') >= 0 or name_index.find(
            'STAGEI-1h') >= 0 or name_index.find('STAGEI_1h') >= 0 or name_index.find(
            'Stage1-0h') >= 0 or name_index.find('Stage1_0h') >= 0 or name_index.find(
            'Stage1-1h') >= 0 or name_index.find('Stage1_1h') >= 0 or name_index.find(
            'StageI-0h') >= 0 or name_index.find('StageI_0h') >= 0 or name_index.find(
            'StageI-1h') >= 0 or name_index.find('StageI_1h') >= 0 or name_index.find(
            'stage1-0h') >= 0 or name_index.find('stage1_0h') >= 0 or name_index.find(
            'stage1-1h') >= 0 or name_index.find('stage1_1h') >= 0 or name_index.find(
            'stageI-0h') >= 0 or name_index.find('stageI_0h') >= 0 or name_index.find(
            'stageI-1h') >= 0 or name_index.find('stageI_1h') >= 0:
            E = 2
        else:
            E = 3

    if name_index in afterCHIR_CD09:

        print('>>>', S_index, '---', name_index, 'I am doing well :::')
        img = cv2.imread(well_image, -1)

        if S in edge:
            b_list = b_list_e
        else:
            b_list = b_list_m

        for i in range(0, len(b_list)):

            N = i + 1

            if S < 21 or (S == 21 and T < 11) or (S == 21 and T == 11 and N < 17):
                return True

            to_img = img[b_list[i][0]:b_list[i][2], b_list[i][1]:b_list[i][3]]
            to_name = name_index + '~N' + str(N) + '.png'

            to_file = os.path.join(main_path, folder_temp, SSS_folder, S_index, to_name)
            if not os.path.exists(os.path.join(main_path, folder_temp)):
                os.makedirs(os.path.join(main_path, folder_temp))
            if not os.path.exists(os.path.join(main_path, folder_temp, SSS_folder)):
                os.makedirs(os.path.join(main_path, folder_temp, SSS_folder))
            if not os.path.exists(os.path.join(main_path, folder_temp, SSS_folder, S_index)):
                os.makedirs(os.path.join(main_path, folder_temp, SSS_folder, S_index))

            cv2.imwrite(to_file, to_img)

            fake_S = (S - 1) * 25 + (N)
            call_matlab_FC(main_path, to_file, E, T, fake_S, result_path='FractalCurving', this_async=False,
                           do_curving_now=False, sleep_s=0)
            time.sleep(3)

    return True


# def call_matlab_template(img_path, this_async=True, sleep_s=0):
#     if not os.path.exists(img_path):
#         print('!ERROR! The img_path does not existed!')
#         return False
#
#     # matlab_engine = matlab.engine.start_matlab(async=True).result()
#     # matlab_engine.test_py_call_m()
#
#     # matlab_engine = matlab.engine.start_matlab(async=True).result()
#     # matlab_results = matlab_engine.test_py_call_m()
#     # print('-----------')
#     # print(matlab_results)
#
#     matlab_engine = matlab.engine.start_matlab()
#     # print(img_path)
#     matlab_call = matlab_engine.Task_Fractal_and_Curving(img_path, async=this_async)
#     print('! finished call -----------')
#     # print(matlab_call.result())
#     time.sleep(sleep_s)
#     # print('after 10s -----------')
#     # print(matlab_call.result())
#
#     # matlab_engine = matlab.engine.start_matlab()
#     # matlab_engine.test_py_call_m()
#
#     # matlab_engine = matlab.engine.start_matlab()
#     # matlab_results = matlab_engine.test_py_call_m()
#     # print('-----------')
#     # print(matlab_results)
#     return True


def call_matlab_FC(main_path, img_path, E, T, S, result_path='FractalCurving', this_async=True, do_curving_now=False,
                   sleep_s=0):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(img_path):
        print('!ERROR! The img_path does not existed!')
        return False
    result_path = os.path.join(main_path, result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    matlab_engine = matlab.engine.start_matlab()
    matlab_call = matlab_engine.Task_Fractal_and_Curving(do_curving_now, main_path, result_path, img_path,
                                                         matlab.int32([E]), matlab.int32([T]), matlab.int32([S]))
    print(matlab_call)
    matlab_engine.quit()
    time.sleep(sleep_s)

    return True


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Lib_Features.py !')

    main_path = r'G:\CD46\PROCESSING'
    after_PGC_do_Fractal(main_path, times=15, sort_function=files_sort_univers, sleep_s=1)

    # main_path = r'T:\CD11'
    # zoom = 1
    # research_stitched_image_elastic_bat(main_path, zoom, files_sort_CD11, make_afterCHIR_CD11_myPGC, do_SSS=False,
    #                                     do_SSSS=True, do_parallel=False, process_number=30)

    # main_path = r'C:\C137\PROCESSING\CD13'
    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD13, make_firstIPS_CD13_myPGC, do_SSS=True,
    #                                     do_SSSS=True, do_parallel=False, process_number=30)

    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD27, core_features_one_image, do_SSS=False,
    #                                     do_SSSS=True, do_parallel=False, process_number=30)

    # main_path = r'C:\C137\PROCESSING\CD09\MyPGC_img'
    # zoom = 0.5
    # research_stitched_image_elastic_bat(main_path, zoom, files_sort_CD09, make_afterCHIR_reseach_CD09, do_SSS=True,
    #                                     do_SSSS=False, do_parallel=False, process_number=30)

    # main_path = r'E:\CD09\PROCESSING\MyPGC_img'
    # zoom = 0.5
    # research_stitched_image_elastic_bat(main_path, zoom, files_sort_CD09, make_afterCHIR_reseach_CD09, do_SSS=True,
    #                                     do_SSSS=False, do_parallel=False, process_number=30)

    # matlab_engine = matlab.engine.start_matlab()
    # matlab_call = matlab_engine.Task_Fractal_and_Curving(False, main_path, result_path, img_path,
    #                                                      matlab.int32([E]), matlab.int32([T]), matlab.int32([S]),
    #                                                      async=this_async)

    # img_path = r'C:\Users\Kitty\Documents\Desktop\Fractal_\test\2018-12-01~I-1_CD13~T1.jpg'
    # call_matlab_FC(main_path, img_path, 1, 1, 1, result_path='FractalCurving', this_async=True, sleep_s=120)

    # main_path = r'D:\CD39\PROCESSING'
    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD39, make_afterCHIR_CD39_myPGC, do_SSS=True,
    #                                     do_SSSS=False, do_parallel=True, process_number=30)

    # pyramid_features(r'C:\Users\Kitty\Desktop\CD22',
    #                  r'C:\Users\Kitty\Desktop\CD22\SSSS_100%\S1\2019-05-18~IPS-1_CD22~T1.png', '',
    #                  result_path='Pyramid')

    # main_path = r'I:\PROCESSING\CD13_0'
    # research_stitched_SSSS_image(main_path, 1.0)

    # merge_all_well_features(r'L:\CD30\PROCESSING', 'Analysis', output_name='Features.csv')
    # merge_all_well_features(r'L:\CD30\PROCESSING', 'Enhanced_Analysis', output_name='Enhanced_Features.csv')
    # merge_all_well_features(r'L:\CD30\PROCESSING', 'MyPGC_Analysis', output_name='MyPGC_Features.csv')

    # main_path = r'D:\PROCESSING\CD11'

    # research_stitched_image(main_path, 1, files_sort_CD26, get_density_save)
    # research_stitched_image(main_path, 1, files_sort_CD13, ext_images_features_high_contrast)

    # research_stitched_image(main_path, 1, files_sort_CD11, get_density_save)

    # main_path = r'C:\C137\PROCESSING\CD13'
    # research_stitched_image(main_path, 1, files_sort_CD13, get_density_perimeter_save)

    # main_path = r'C:\Users\Kitty\Documents\Desktop\test1'
    # input_avi = r'C:\Users\Kitty\Documents\Desktop\test1\SSS_S14.avi'
    # video_density_flow_test(main_path, input_avi)

    # main_path = r'F:\PROCESSING\CD26'
    # research_stitched_image(main_path, 1, files_sort_CD26, get_density_perimeter_save)
    # main_path = r'M:\CD27\PROCESSING'
    # research_stitched_image(main_path, 1, files_sort_CD27, get_density_perimeter_save)
    # main_path = r'D:\CD39\PROCESSING'
    # research_stitched_image(main_path, 1, files_sort_CD39, get_density_perimeter_save)
    # img_path = r'C:\Users\Kitty\Documents\Desktop\Fractal\CD13_S44\2018-12-01~I-1_CD13~T4.jpg'
    # img = ImageData(img_path,show=False)
    # result = img.getContours(show=False)
    # # print(type(result))
    # # print(result)
    # cv2.imwrite(r'C:\Users\Kitty\Documents\Desktop\Fractal\123.png',result)

    # path = r'C:\Users\Kitty\Documents\Desktop\Fractal\CD13_S44'
    # result_path = r'C:\Users\Kitty\Documents\Desktop\Fractal\CD13_S44_C_1'
    #
    # dir_list = os.listdir(path)
    # for each_f in dir_list:  # S1 to S96
    #     each_file = os.path.join(path, each_f)
    #     img = ImageData(each_file)
    #     each_result = img.getContours()
    #     each_result_file = os.path.join(result_path, each_f)
    #     cv2.imwrite(each_result_file, each_result)

    # each_file = r'C:\Users\Kitty\Documents\Desktop\Fractal\CD13_S44\2018-12-01~I-1_CD13~T10.jpg'
    # img = ImageData(each_file)
    # img.getDensity(show=True)
    # img.getPerimeter(show=True)
    # img.getContours(show=True)

    # main_path = r'M:\CD27\PROCESSING'
    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD27, core_features_one_image, do_SSS=False,
    #                                     do_SSSS=True, do_parallel=False, process_number=30)

    # main_path = r'D:\CD39\PROCESSING'
    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD39, core_features_one_image, do_SSS=False,
    #                                     do_SSSS=True, do_parallel=False, process_number=30)

    # main_path = r'H:\CD41\PROCESSING'
    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD41, core_features_one_image, do_SSS=False,
    #                                     do_SSSS=True, do_parallel=False, process_number=30)

    # main_path = r'D:\CD39\PROCESSING'
    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD39, make_afterCHIR_CD39_myPGC, do_SSS=True,
    #                                     do_SSSS=False, do_parallel=True, process_number=30)

    # main_path = r'H:\CD41\PROCESSING'
    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD41, make_afterCHIR_CD41_myPGC, do_SSS=False,
    #                                     do_SSSS=True, do_parallel=True, process_number=30)
