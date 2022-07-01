import os
import time
import shutil
# import random
# from collections import Iterable
import numpy as np
import pandas as pd
from PIL import Image
import cv2
# import matplotlib.image as pltimg
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from Lib_Class import ImageData
# from sklearn.decomposition import PCA
# from sklearn import manifold
# import skimage
from Lib_Tiles import return_96well_25_Tiles, return_24well_Tiles
from Lib_Sort import files_sort_CD09, files_sort_CD13, files_sort_CD26, files_sort_CD27, files_sort_CD42


def get_time():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def prevent_disk_died(main_path, img_file):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(img_file):
        print('!ERROR! The img_file does not existed!')
        return False

    while True:
        temp = os.listdir(main_path)
        img = any_to_image(img_file)
        get_time()
        time.sleep(30)


def any_to_image(img):
    # core methods
    # image pre processing, can processing any image form to np.ndarray format
    # input img can be path str
    # output img is cv2 np.ndarray (colored or original)!

    if type(img) is str:
        if not os.path.exists(img):
            print('!ERROR! The image path does not existed!')
            return None
        try:
            img = cv2.imread(img, cv2.IMREAD_COLOR)  # BGR .shape=(h,w,3)
        except:
            print('cv2.imread error! ')
        img = np.uint8(img)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif type(img) is np.ndarray:
        img = np.uint8(img)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = img[:, :, 0:3]
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            # img_gray = img
            print('!NOTICE! The input image is gray!')
            # pass
        else:
            print('!ERROR! The image shape error!')
            return None
    else:
        print('!ERROR! Please input correct CV2 image file or file path!')
        return None

    return img


def image_to_gray(img):
    # core methods
    # can transform colored image to gray:  (.shap=(h,w))
    # input img can be path str or np.ndarray
    # output img is cv2 np.ndarray gray!

    img = any_to_image(img)
    if img is None:
        return None

    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img_gray = img
    else:
        print('!ERROR! The image shape error!')
        return None

    return img_gray


def to_8bit(x):
    x = np.where(x < 0, 0, x)
    x = np.where(x > 255, 255, x)
    return x


def to_8bit_2(x):
    x = np.where(x < 0, 0, x)
    x = np.where(x > 255 * 2, 255, x)
    return x


def is_number(s):
    # if it's '1.23' or 1.23 , it will return 1.23
    try:
        float(s)
        return float(s)
    except ValueError:
        return np.NaN


def is_float(x):
    # if it's '1.23' str float, it will return np.NaN
    if isinstance(x, float) or isinstance(x, int):
        return float(x)
    else:
        return np.NaN


def get_cpu_load():
    """ Returns a list CPU Loads"""
    cmd = "wmic cpu get LoadPercentage"
    response = os.popen(cmd).read().strip().split("\n")
    # print(response)
    return int(response[-1])


def get_cpu_python_process():
    cmd = "wmic process where name=\"python.exe\" list brief"
    response = os.popen(cmd).read().strip().split("\n\n")
    # print(response)
    return len(response) - 1


def python_busy_sleep(max_p):
    MAX_PYTHON_process_number = max_p
    temp_p_number = get_cpu_python_process()
    print('!Notice! : $$$---Python Number: ', temp_p_number, '---$$$')
    if temp_p_number > MAX_PYTHON_process_number:
        print('!Notice! : Python process number is more than ', MAX_PYTHON_process_number, ' Now sleep!')
        time.sleep(10)
        while True:
            if get_cpu_python_process() >= MAX_PYTHON_process_number:
                time.sleep(10)
            else:
                break
    return True


def scan_main_path(main_path):
    # find all existed experiment folder
    # input: main_path: the folder
    # output: result: main_path, date_path, name_path
    result = []
    date_paths = os.listdir(main_path)
    for date_path in date_paths:
        if os.path.isdir(os.path.join(main_path, date_path)):
            name_paths = os.listdir(os.path.join(main_path, date_path))
            for name_path in name_paths:
                this_path_date_name = os.path.join(main_path, date_path, name_path)
                if os.path.isdir(this_path_date_name):
                    this_path_date_names = os.listdir(this_path_date_name)
                    for B_path in this_path_date_names:
                        if B_path == 'B' or B_path == 'B=0':
                            result.append(os.path.join(main_path, date_path, name_path))
                            break
    return result


def scan_dish_margin(main_path):
    margin = ()
    if os.path.exists(os.path.join(main_path, 'dish_margin.txt')):
        f = open(os.path.join(main_path, 'dish_margin.txt'))
        f.seek(0)
        margin = f.read().split(',')
        margin = tuple(map(int, margin))
    return margin


def get_specific_Mpath(path, B, T, S, Z, C):
    # get only carl zeiss auto export image path (ZEN2.3 ZEN2.5)
    # B, T, S, Z, C from 1
    if not os.path.exists(path):
        return None

    # (ZEN2.3)
    this_path = os.path.join(path, 'B=' + str(B - 1), 'T=' + str(T - 1), 'S=' + str(S - 1), 'Z=' + str(Z - 1),
                             'C=' + str(C - 1))
    if os.path.exists(this_path):
        return this_path

    if T == 1:
        this_path = os.path.join(path, 'B=' + str(B - 1), 'S=' + str(S - 1), 'Z=' + str(Z - 1), 'C=' + str(C - 1))
        if os.path.exists(this_path):
            return this_path

    if Z == 1:
        this_path = os.path.join(path, 'B=' + str(B - 1), 'T=' + str(T - 1), 'S=' + str(S - 1), 'C=' + str(C - 1))
        if os.path.exists(this_path):
            return this_path

    if T == 1 & Z == 1:
        this_path = os.path.join(path, 'B=' + str(B - 1), 'S=' + str(S - 1), 'C=' + str(C - 1))
        if os.path.exists(this_path):
            return this_path

    # (ZEN2.5)
    if B == 1:
        this_path = os.path.join(path, 'B', 'T=' + str(T - 1), 'S=' + str(S - 1), 'Z=' + str(Z - 1), 'C=' + str(C - 1))
        if os.path.exists(this_path):
            return this_path

        if T == 1:
            this_path = os.path.join(path, 'B', 'S=' + str(S - 1), 'Z=' + str(Z - 1), 'C=' + str(C - 1))
            if os.path.exists(this_path):
                return this_path

        if Z == 1:
            this_path = os.path.join(path, 'B', 'T=' + str(T - 1), 'S=' + str(S - 1), 'C=' + str(C - 1))
            if os.path.exists(this_path):
                return this_path

        if T == 1 & Z == 1:
            this_path = os.path.join(path, 'B', 'S=' + str(S - 1), 'C=' + str(C - 1))
            if os.path.exists(this_path):
                return this_path

    return None


def get_specific_image(path, B, T, S, Z, C, M):
    # get only carl zeiss auto export (ZEN2.3 ZEN2.5)
    if os.path.exists(path):
        this_path = get_specific_Mpath(path, B, T, S, Z, C)
        if os.path.exists(this_path):
            this_files = os.listdir(this_path)
            for this_file in this_files:
                if int(this_file.split('_M')[-1].split('_')[0]) == M - 1:
                    return os.path.join(this_path, this_file)
    return None


def get_CZI_image(path, B, T, S, Z, C, M):
    # get carl zeiss auto export OR image exported images (ZEN2.3 ZEN2.5)
    # !!!B, T, S, Z, C, M!!! is from 1!!!
    # path is D:\processing\CD16\2019-01-10\I-1_CD16
    # file is B=0\T=0\S=0\Z=2\C=0\IPS_CD13_S0000(A1-A1)_T000000_Z0000_C00_M0000_ORG.jpg
    # OR
    # path is D:\processing\CD16\image_exported\2018-09-03~I-1_CD09
    # file is 2018-09-03~I-1_CD09~s01t01z1m001.png
    # file is 2018-09-03~I-1_CD09~info.xml
    # file is 2018-09-03~I-1_CD09~meta.xml
    # output:
    # [0,1,2,3,4,5]
    # [img_path,'S1','2018-09-03','I-1_CD09','T1','Z1']
    if not os.path.exists(path):
        return None

    image_path = None
    AE_path = get_specific_Mpath(path, B, T, S, Z, C)

    if (AE_path is not None) and os.path.exists(AE_path):
        this_path = AE_path
        for this_file in os.listdir(this_path):
            if this_file.find('_M') == -1:
                continue
            if int(this_file.split('_M')[-1].split('_')[0]) == M - 1:
                image_path = os.path.join(this_path, this_file)
                t_xep_namp = os.path.split(path)  # D:\processing\CD16\2019-01-10    I-1_CD16
                date_name = os.path.split(t_xep_namp[0])[1]  # D:\processing\CD16         2019-01-10
                exp_name = t_xep_namp[1]
                break
    else:
        this_path = path  # IEXP_path
        # look_for = 's' + str(S) + 't' + str(T) + 'z' + str(Z) + 'm' + str(M)
        this_b = -1
        this_t = -1
        this_s = -1
        this_z = -1
        this_c = -1
        this_m = -1
        for this_file in os.listdir(this_path):
            if this_file.find('~') != -1:  # find ~  r'2018-11-06_IPS_0_CD11_s01t36z1m01_ORG.png'

                if this_file.find('_ORG.') != -1:
                    this_name = this_file.split('_ORG.')[0].split('~')[-1]
                elif this_file.find('.') != -1:
                    this_name = this_file.split('.')[0].split('~')[-1]  # this_name can be r's01t36z1m01' info meta
                else:
                    return None

            elif this_file.find('_') != -1:  # find _

                if this_file.find('_ORG.') != -1:
                    this_name = this_file.split('_ORG.')[0].split('_')[-1]
                    # this_name = this_file.split('_')[-2].split('_ORG.')[0]
                elif this_file.find('.') != -1:
                    # this_name = this_file.split('~')[-1].split('.')[0]  # this_name can be r's01t36z1m01' info meta
                    this_name = this_file.split('.')[0].split('_')[-1]
                else:
                    return None
            else:
                return None

            # print(this_name)
            # this_name can be s01t01z1m001   s01z1m001   s01t01m001   s01c01m01
            if this_name.find('s') != -1:  # if found s01
                if this_name.find('t') != -1:  # s01t01
                    this_s = int(this_name.split('t')[0].split('s')[1])
                    if this_name.find('z') != -1:  # s01t01z01
                        this_t = int(this_name.split('z')[0].split('t')[1])
                        if this_name.find('c') != -1:  # s01t01z01c01
                            this_z = int(this_name.split('c')[0].split('z')[1])
                            if this_name.find('m') != -1:  # s01t01z01c01m01
                                this_c = int(this_name.split('m')[0].split('c')[1])
                                this_m = int(this_name.split('m')[1])
                            else:  # s01t01z01c01
                                this_c = int(this_name.split('c')[1])
                                this_m = 1
                        elif this_name.find('m') != -1:  # s01t01z01m01
                            this_z = int(this_name.split('m')[0].split('z')[1])
                            this_c = 1
                            this_m = int(this_name.split('m')[1])
                        else:  # s01t01z01
                            this_z = int(this_name.split('z')[1])
                            this_c = 1
                            this_m = 1
                    elif this_name.find('c') != -1:  # s01t01c01
                        this_t = int(this_name.split('c')[0].split('t')[1])
                        this_z = 1
                        if this_name.find('m') != -1:  # s01t01c01m01
                            this_c = int(this_name.split('m')[0].split('c')[1])
                            this_m = int(this_name.split('m')[1])
                        else:  # s01t01c01
                            this_c = int(this_name.split('c')[1])
                            this_m = 1
                    elif this_name.find('m') != -1:  # s01t01m01
                        this_t = int(this_name.split('m')[0].split('t')[1])
                        this_z = 1
                        this_c = 1
                        this_m = int(this_name.split('m')[1])
                    else:  # s01t01
                        this_t = int(this_name.split('t')[1])
                        this_z = 1
                        this_c = 1
                        this_m = 1
                elif this_name.find('z') != -1:  # s01z01
                    this_s = int(this_name.split('z')[0].split('s')[1])
                    this_t = 1
                    if this_name.find('c') != -1:  # s01z01c01
                        this_z = int(this_name.split('c')[0].split('z')[1])
                        if this_name.find('m') != -1:  # s01z01c01m01
                            this_c = int(this_name.split('m')[0].split('c')[1])
                            this_m = int(this_name.split('m')[1])
                        else:  # s01z01c01
                            this_c = int(this_name.split('c')[1])
                            this_m = 1
                    elif this_name.find('m') != -1:  # s01z01m01
                        this_z = int(this_name.split('m')[0].split('z')[1])
                        this_c = 1
                        this_m = int(this_name.split('m')[1])
                    else:  # s01z01
                        this_z = int(this_name.split('z')[1])
                        this_c = 1
                        this_m = 1
                elif this_name.find('c') != -1:  # s01c01
                    this_s = int(this_name.split('c')[0].split('s')[1])
                    this_t = 1
                    this_z = 1
                    if this_name.find('m') != -1:  # s01c01m01
                        this_c = int(this_name.split('m')[0].split('c')[1])
                        this_m = int(this_name.split('m')[1])
                    else:  # s01c01
                        this_c = int(this_name.split('c')[1])
                        this_m = 1
                elif this_name.find('m') != -1:  # s01m01
                    this_s = int(this_name.split('m')[0].split('s')[1])
                    this_t = 1
                    this_z = 1
                    this_c = 1
                    this_m = int(this_name.split('m')[1])
                else:  # s01
                    this_s = int(this_name.split('s')[1])
                    this_t = 1
                    this_z = 1
                    this_c = 1
                    this_m = 1
            else:
                # do not find 's'
                # the info.xml or meta.xml files
                pass

            # print(B, ' == 1 and ', T, ' == ', this_t, ' and ', S, ' == ', this_s, ' and ', Z, ' == ', this_z, ' and ',
            #       C, ' == ', this_c, ' and ', M, ' == ', this_m)
            if B == 1 and T == this_t and S == this_s and Z == this_z and C == this_c and M == this_m:
                image_path = os.path.join(this_path, this_file)
                t_xep_namp = os.path.split(path)[1]  # D:\processing\CD16\image_exported\   2018-09-03~I-1_CD09
                try:
                    # 2018-11-13_1200_II-3_CD11_s01t24z1m01_ORG.tif
                    this_file_head = this_file.split(this_name)[0]
                    # 2018-11-13_1200_II-3_CD11_  2018-11-13~1200_II-3_CD11~
                    if this_file_head.find('~') != -1:
                        date_name = this_file_head.split('~')[0]
                        exp_name = this_file_head.split('~')[-2]
                    elif this_file_head.find('_') != -1:
                        date_name = this_file_head.split('_')[0]
                        exp_name = this_file_head.split('_')[-2]
                    else:
                        date_name = None
                        exp_name = None
                except BaseException as e:
                    print('!ERROR! ', e)
                    date_name = None
                    exp_name = None
                else:
                    pass
                finally:
                    pass
                break

    # print(image_path)
    if image_path is None:
        return None
    else:  # [img_path, 'S1', '2018-09-03', 'I-1_CD09', 'T1', 'Z1', 'C1', 'M1']
        return [image_path, 'S' + str(S), date_name, exp_name, 'T' + str(T), 'Z' + str(Z), 'C' + str(C), 'M' + str(M)]


def get_ImageVar(input):
    # get the gray image focus sharpness
    # input can be image path
    # or np.ndarray cv2 image file
    # output is imageVar

    img2gray = image_to_gray(input)
    if img2gray is None:
        return None
    # if type(input) is str:
    #     if not os.path.exists(input):
    #         print('!ERROR! The image path does not existed!')
    #         return None
    #     img2gray = cv2.imread(input, 0)
    # elif type(input) is np.ndarray:
    #     image = input
    # else:
    #     print('!ERROR! Please input correct CV2 image file or file path!')
    #     return None
    # img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    except BaseException as e:
        print('!ERROR! ', e)
        imageVar = 0
    else:
        # print('get_ImageVar(', input, ')')
        pass
    finally:
        pass

    return imageVar


def trans_blur(img, ksize=13):
    if type(img) is np.ndarray:
        image = img
    else:
        # print('!ERROR! Please input correct CV2 image!')
        image = any_to_image(img)
        # return None

    return cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=0, sigmaY=0)


def trans_CLAHE(img, tileGridSize=16):
    if type(img) is np.ndarray:
        image = img
    else:
        # print('!ERROR! Please input correct CV2 image!')
        image = any_to_image(img)
        # return None

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(tileGridSize, tileGridSize))
    if len(image.shape) == 3:
        out = np.array(image)
        for i in range(image.shape[-1]):
            out[:, :, i] = clahe.apply(image[:, :, i])
    elif len(image.shape) == 2:
        out = clahe.apply(image)

    return out


def trans_Unsharp_Masking(img, ksize=13, k=1.0, o=1):
    if type(img) is np.ndarray:
        image = img
    else:
        # print('!ERROR! Please input correct CV2 image!')
        image = any_to_image(img)
        # return None

    img_blur = cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=0, sigmaY=0)

    # np.uint8(np.array(list(map(to_8bit, image + k * (image - img_blur)))))
    enhanced = o * image + k * (image - img_blur)
    # enhanced=to_8bit_2(enhanced)
    enhanced = np.uint8(enhanced)

    return enhanced


def trans_myPGC(img, ksize=111, k=1.0, o=0):
    image = any_to_image(img)
    blur = cv2.GaussianBlur(image, (ksize, ksize), 0)

    enhanced = o * image + k * (image - blur)
    enhanced = np.uint8(enhanced)

    return enhanced


def trans_gamma(img, gamma=2.2):
    # input img is np.ndarray cv2 image file
    # gamma = 2.2
    # output np.ndarray
    if type(img) is np.ndarray:
        image = img
    else:
        # print('!ERROR! Please input correct CV2 image!')
        image = any_to_image(img)
        # return None

    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(image, gamma_table)


def trans_line(img, low_cut, hight_cut):
    # input img is np.ndarray cv2 image file
    # low_cut,hight_cut: line Histogram between: 0~1
    # output np.ndarray
    if type(img) is np.ndarray:
        image = img
    else:
        # print('!ERROR! Please input correct CV2 image!')
        # return None
        image = any_to_image(img)

    line_table = []
    for i in range(256):
        if i <= low_cut * 255:
            line_table.append(0)
        elif i >= hight_cut * 255:
            line_table.append(255)
        else:
            line_table.append((i - 255 * low_cut) / (hight_cut - low_cut))

    line_table = np.round(np.array(line_table)).astype(np.uint8)

    return cv2.LUT(image, line_table)


def trans_sub_mode(img):
    # input img is np.ndarray cv2 image file
    # low_cut,hight_cut: line Histogram between: 0~1
    # output np.ndarray
    if type(img) is np.ndarray:
        image = img
    else:
        # print('!ERROR! Please input correct CV2 image!')
        # return None
        image = any_to_image(img)

    line_table = [i for i in range(256)]
    mode = np.argmax(np.bincount(image.flatten()))
    line_table[mode] = 0
    line_table = np.round(np.array(line_table)).astype(np.uint8)

    return cv2.LUT(image, line_table)


def color_image_combination(img_R=None, img_G=None, img_B=None):
    if img_R is None and img_G is None and img_B is None:
        print('!ERROR! The input image must have a channel!')
        return None
    if img_R is not None:
        img_R = image_to_gray(img_R)
        shape = img_R.shape
    if img_G is not None:
        img_G = image_to_gray(img_G)
        shape = img_G.shape
    if img_B is not None:
        img_B = image_to_gray(img_B)
        shape = img_B.shape

    result = np.zeros((*shape, 3), dtype=np.uint8)

    if img_R is not None:
        result[:, :, 2] = np.uint8(to_8bit(result[:, :, 2] + img_R))
    if img_G is not None:
        result[:, :, 1] = np.uint8(to_8bit(result[:, :, 1] + img_G))
        result[:, :, 0] = np.uint8(to_8bit(result[:, :, 0] + (40 / 255) * img_G))
    if img_B is not None:
        result[:, :, 0] = np.uint8(to_8bit(result[:, :, 0] + img_B))
        result[:, :, 1] = np.uint8(to_8bit(result[:, :, 1] + (160 / 255) * img_B))

    return result


def image_Histogram_Equalization(img_file, to_file, show_hist=False):
    # only do image_Histogram_Equalization
    if not os.path.exists(img_file):
        print('!ERROR! The img_file does not existed!')
        return False

    img = cv2.imread(img_file, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow(img_file, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if show_hist:
        plt.figure('hist')
        arr = img.flatten()
        n, bins, patches = plt.hist(arr, bins=256, normed=1, edgecolor='None', facecolor='black', alpha=0.5)
        plt.show()
        plt.close()

    img = cv2.equalizeHist(img)

    if show_hist:
        plt.figure('after enhancement hist')
        arr = img.flatten()
        n, bins, patches = plt.hist(arr, bins=256, normed=1, edgecolor='None', facecolor='black', alpha=0.5)
        plt.show()
        plt.close()

    # img = gamma_trans(img, gamma=2)
    # cv2.imshow('after enhancement', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(to_file, img)

    return True


def image_treatment_toGray(input_img, show=False):
    # image CLAHE enhancement
    # input is numpy image or file path
    # outout is numpy Gray image

    if type(input_img) is str:
        if not os.path.exists(input_img):
            print('!ERROR! The image path does not existed!')
            return None
        # print(input_img)
        img = cv2.imread(input_img, 0)
        img = np.uint8(img)
    elif type(input_img) is np.ndarray:
        img = np.uint8(input_img)
    else:
        print('!ERROR! Please input correct CV2 image file or file path!')
        return None

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img_gray = img
        # print('!NOTICE! The input image is gray!')
    else:
        print('!ERROR! The image shape error!')
        return None

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
    img_out = clahe.apply(img_gray)

    if show:
        cv2.imshow('ori_image', img)
        cv2.waitKey(0)
        cv2.imshow('out_image', img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_out


def image_retreatment_toGray(input_img, show=False, CLAHE=True, auto_curve=False, Unsharp_Masking=False, myPGC=False):
    # for auto stitching aoto enhance
    # image enhancement pre treatment
    # input is numpy image or file path
    # outout is numpy Gray image

    # show = True

    if type(input_img) is str:
        if not os.path.exists(input_img):
            print('!ERROR! The image path does not existed!')
            return None
        # print(input_img)
        img = cv2.imread(input_img, 0)
        img = np.uint8(img)
    elif type(input_img) is np.ndarray:
        img = np.uint8(input_img)
    else:
        print('!ERROR! Please input correct CV2 image file or file path!')
        return None

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img_gray = img
        # print('!NOTICE! The input image is gray!')
    else:
        print('!ERROR! The image shape error!')
        return None

    if CLAHE:
        img_gray = trans_CLAHE(img_gray, tileGridSize=16)

    if auto_curve:
        theta = 4.8
        gamma = 1.08
        img_array = img_gray.flatten()
        img_gray = trans_line(img_gray, (np.mean(img_array) - theta * np.std(img_array)) / 255,
                              (np.mean(img_array) + theta * np.std(img_array)) / 255)
        img_gray = trans_gamma(img_gray, gamma=gamma)

    if Unsharp_Masking:
        img_gray = trans_Unsharp_Masking(img_gray, ksize=13, k=1.0, o=1)

    if myPGC:
        img_gray = trans_myPGC(img_gray, ksize=111, k=1.0, o=0)

    if show:
        cv2.imshow('ori_image', img)
        cv2.waitKey(0)
        cv2.imshow('retreatment_image', img_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # if len(img.shape) == 3:
    #     return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # elif len(img.shape) == 2:
    #     return img_gray
    return img_gray


def image_enhancement(img_file, to_file, show_hist=False):
    # image enhancement pre treatment
    pass


def image_my_enhancement_experience(img_file, to_file):
    # only do line_trans and gamma_trans
    if not os.path.exists(img_file):
        print('!ERROR! The img_file does not existed!')
        return False

    img = cv2.imread(img_file, -1)
    img = trans_line(img, 0.125, 0.875)
    img = trans_gamma(img, gamma=1.95)
    cv2.imwrite(to_file, img)

    return True


def image_my_enhancement(img_file, to_file, show_hist=False, cut_off=3, gamma=1):
    # auto adjust line_trans
    if not os.path.exists(img_file):
        print('!ERROR! The img_file does not existed!')
        return False

    img = cv2.imread(img_file, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_array = img.flatten()

    if show_hist:
        plt.figure('hist')
        n, bins, patches = plt.hist(img_array, bins=256, normed=1, edgecolor='None', facecolor='black', alpha=0.5)
        hist_y = mlab.normpdf(bins, np.mean(img_array), np.std(img_array))
        plt.plot(bins, hist_y, 'r--')
        plt.hlines(np.max(hist_y), np.min(bins), np.max(bins), colors='c', linestyles='dashed')
        plt.hlines(np.max(hist_y) / 100, np.min(bins), np.max(bins), colors='c', linestyles='dashed')
        plt.vlines(np.mean(img_array) - np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) - 2 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) - 3 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + 2 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + 3 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.show()
        plt.close()

    img = trans_line(img, (np.mean(img_array) - cut_off * np.std(img_array)) / 255,
                     (np.mean(img_array) + cut_off * np.std(img_array)) / 255)

    if gamma != 1:
        img = trans_gamma(img, gamma=gamma)

    if show_hist:
        img_array = img.flatten()
        plt.figure('after enhancement hist')
        n, bins, patches = plt.hist(img_array, bins=256, normed=1, edgecolor='None', facecolor='black', alpha=0.5)
        hist_y = mlab.normpdf(bins, np.mean(img_array), np.std(img_array))
        plt.plot(bins, hist_y, 'r--')
        plt.hlines(np.max(hist_y), np.min(bins), np.max(bins), colors='c', linestyles='dashed')
        plt.hlines(np.max(hist_y) / 100, np.min(bins), np.max(bins), colors='c', linestyles='dashed')
        plt.vlines(np.mean(img_array) - np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) - 2 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) - 3 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + 2 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + 3 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.show()
        plt.close()

    cv2.imwrite(to_file, img)

    return True


def image_my_enhancement_sub_mode(img_file, to_file, show_hist=False):
    # auto adjust line_trans
    if not os.path.exists(img_file):
        print('!ERROR! The img_file does not existed!')
        return False

    img = cv2.imread(img_file, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_array = img.flatten()

    if show_hist:
        plt.figure('hist')
        n, bins, patches = plt.hist(img_array, bins=256, normed=1, edgecolor='None', facecolor='black', alpha=0.5)
        hist_y = mlab.normpdf(bins, np.mean(img_array), np.std(img_array))
        plt.plot(bins, hist_y, 'r--')
        plt.hlines(np.max(hist_y), np.min(bins), np.max(bins), colors='c', linestyles='dashed')
        plt.hlines(np.max(hist_y) / 100, np.min(bins), np.max(bins), colors='c', linestyles='dashed')
        plt.vlines(np.mean(img_array) - np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) - 2 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) - 3 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + 2 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + 3 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.show()
        plt.close()

    img = trans_line(img, (np.mean(img_array) - 2 * np.std(img_array)) / 255,
                     (np.mean(img_array) + 2 * np.std(img_array)) / 255)
    img = trans_gamma(img, gamma=1.85)

    if show_hist:
        img_array = img.flatten()
        plt.figure('after enhancement hist')
        n, bins, patches = plt.hist(img_array, bins=256, normed=1, edgecolor='None', facecolor='black', alpha=0.5)
        hist_y = mlab.normpdf(bins, np.mean(img_array), np.std(img_array))
        plt.plot(bins, hist_y, 'r--')
        plt.hlines(np.max(hist_y), np.min(bins), np.max(bins), colors='c', linestyles='dashed')
        plt.hlines(np.max(hist_y) / 100, np.min(bins), np.max(bins), colors='c', linestyles='dashed')
        plt.vlines(np.mean(img_array) - np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) - 2 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) - 3 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + 2 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.vlines(np.mean(img_array) + 3 * np.std(img_array), 0, np.max(hist_y), colors='b', linestyles='dashed')
        plt.show()
        plt.close()

    img = trans_sub_mode(img)
    cv2.imwrite(to_file, img)

    return True


def image_my_PGC(img_file, to_file, show_hist=False):
    # my processing test
    # path to path
    if not os.path.exists(img_file):
        print('!ERROR! The img_file does not existed!')
        return False

    img = cv2.imread(img_file, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_array = img.flatten()
    # img_2std = line_trans(img, (np.mean(img_array) - 2 * np.std(img_array)) / 255,
    #                       (np.mean(img_array) + 2 * np.std(img_array)) / 255)

    blur = cv2.GaussianBlur(img, (111, 111), 0)
    img_1 = np.abs(img - blur)
    # blur = cv2.GaussianBlur(img, (301, 301), 0)
    # img_2 = np.abs(img - blur)

    # img = np.min([img.flatten(), img_1.flatten()], axis=0).reshape(img.shape)
    # img=(img*img_1)/255
    # for i in range(len(img.flatten())):
    #     if

    cv2.imwrite(to_file, img_1)

    return True


def get_AE_density(this_F_index, this_B, this_T, this_S, this_Z, this_C, this_M):
    this_image_path = get_specific_image(this_F_index, this_B, this_T, this_S, this_Z, this_C, this_M)  # Z=2; C=1;
    if this_image_path is not None and os.path.exists(this_image_path):
        this_image = ImageData(this_image_path, 0)
        # print('  !!!success!!!  ', end='')
        # this_image.getDensity()
        return this_image.getDensity()
    else:
        # print('  !!!ZEN Export error!!!  ', end='')
        return None


def get_img_density(imgae_path):
    if imgae_path is not None and os.path.exists(imgae_path):
        this_image = ImageData(imgae_path, 0)
        # this_image.getDensity()
        return this_image.getDensity()
    else:
        print('Image path not exists!')
        return None


def image_resize(main_path, zoom_origin, zoom_in, this_index_path=None):
    # using the zoom_origin resize to zoom_in size
    # input:
    # main_path: is the main folder
    # this_index_path: the folder to processing, can be '' to processing all folders
    # zoom_origin: exp: 1:100%
    # zoom_in: is a float or a float list exp: 0.3 or [0.3, 0.2, 0.1]
    # output: True or False
    # using exp:
    # index_path = []
    # index_path.append(r'D:\PROCESSING\CD16\2019-01-13\IPS_CD16')
    # index_path.append(r'D:\PROCESSING\CD16\2019-01-14\I-1_CD16')
    # image_resize(args.main_path, index_path, 1, 0.3)

    prefix = []
    prefix.append('SSS')
    prefix.append('SSSS')

    if not os.path.exists(main_path):
        print('The main_path is not existed!')
        return False

    zoom_origin_str = "%.0f%%" % (zoom_origin * 100)

    for p in prefix:
        p_path_o = os.path.join(main_path, p + '_' + zoom_origin_str)
        if p == 'SSS' and (not os.path.exists(p_path_o)):
            print('The origin zoom is not existed!')
            return False

    if not isinstance(zoom_in, list):
        zoom_in = [zoom_in]

    if (this_index_path is None) or this_index_path == '':
        for i_zoom in zoom_in:
            if (zoom_origin < i_zoom):
                print('The origin zoom is less than zoom_in !')
                next
            i_zoom_str = "%.0f%%" % (i_zoom * 100)
            for p in prefix:
                p_path_o = os.path.join(main_path, p + '_' + zoom_origin_str)
                if os.path.exists(p_path_o):
                    p_path_d = os.path.join(main_path, p + '_' + i_zoom_str)
                    if not os.path.exists(p_path_d):
                        os.makedirs(p_path_d)
                    this_S_folders = os.listdir(p_path_o)
                    for this_S_folder in this_S_folders:  # this_S_folder is S1 to S96
                        this_dS_folder_path = os.path.join(p_path_d, this_S_folder)
                        if not os.path.exists(this_dS_folder_path):
                            os.makedirs(this_dS_folder_path)
                        this_images = os.listdir(os.path.join(p_path_o, this_S_folder))
                        for this_image_name in this_images:
                            this_o_image_path = os.path.join(p_path_o, this_S_folder, this_image_name)
                            d_img_path = os.path.join(this_dS_folder_path, this_image_name)

                            this_o_img = cv2.imread(this_o_image_path, -1)
                            d_img = cv2.resize(this_o_img, (0, 0), fx=i_zoom / zoom_origin, fy=i_zoom / zoom_origin,
                                               interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(d_img_path, d_img)

                            # this_o_img = Image.open(this_o_image_path)
                            # w = round(i_zoom * (this_o_img.width / zoom_origin))
                            # h = round(i_zoom * (this_o_img.height / zoom_origin))
                            # d_img = this_o_img.resize((w, h), Image.ANTIALIAS)
                            # d_img.save(d_img_path)
    else:
        if not isinstance(this_index_path, list):
            this_index_path = [this_index_path]
        selected_img_prefix_list = []
        for each_this_index_path in this_index_path:
            t_xep_namp = os.path.split(each_this_index_path)  # D:\processing\CD16\2019-01-10    I-1_CD16
            t_date = os.path.split(t_xep_namp[0])  # D:\processing\CD16         2019-01-10
            selected_img_prefix_list.append(t_date[1] + '~' + t_xep_namp[1] + '~')  # 2019-01-10~I-1_CD16~
        for i_zoom in zoom_in:
            if (zoom_origin < i_zoom):
                print('The origin zoom is less than zoom_in !')
                next
            i_zoom_str = "%.0f%%" % (i_zoom * 100)
            for p in prefix:
                p_path_o = os.path.join(main_path, p + '_' + zoom_origin_str)
                if os.path.exists(p_path_o):
                    p_path_d = os.path.join(main_path, p + '_' + i_zoom_str)
                    if not os.path.exists(p_path_d):
                        os.makedirs(p_path_d)
                    this_S_folders = os.listdir(p_path_o)
                    for this_S_folder in this_S_folders:  # this_S_folder is S1 to S96
                        this_dS_folder_path = os.path.join(p_path_d, this_S_folder)
                        if not os.path.exists(this_dS_folder_path):
                            os.makedirs(this_dS_folder_path)
                        this_images = os.listdir(os.path.join(p_path_o, this_S_folder))
                        for this_image_name in this_images:
                            t_split = this_image_name.split('~')  # 2018-12-03   I-2_CD13   T20.jpg
                            t_name = t_split[0] + '~' + t_split[1] + '~'  # 2018-12-03~I-2_CD13~
                            if t_name in selected_img_prefix_list:
                                this_o_image_path = os.path.join(p_path_o, this_S_folder, this_image_name)
                                d_img_path = os.path.join(this_dS_folder_path, this_image_name)

                                this_o_img = cv2.imread(this_o_image_path, -1)
                                d_img = cv2.resize(this_o_img, (0, 0), fx=i_zoom / zoom_origin, fy=i_zoom / zoom_origin,
                                                   interpolation=cv2.INTER_NEAREST)
                                cv2.imwrite(d_img_path, d_img)

                                # this_o_img = Image.open(this_o_image_path)
                                # w = round(i_zoom * (this_o_img.width / zoom_origin))
                                # h = round(i_zoom * (this_o_img.height / zoom_origin))
                                # d_img = this_o_img.resize((w, h), Image.ANTIALIAS)
                                # d_img.save(d_img_path)
    return True


def image_folder_resize(main_path, zoom_origin, zoom_in=None, pixel=(1000, 1000)):
    # using the zoom_origin resize to zoom_in size
    # input:

    prefix = []
    prefix.append('SSSS')
    prefix.append('SSS')

    if not os.path.exists(main_path):
        print('The main_path is not existed!')
        return False

    zoom_origin_str = "%.0f%%" % (zoom_origin * 100)

    for p in prefix:
        p_path_o = os.path.join(main_path, p + '_' + zoom_origin_str)
        if p == 'SSS' and (not os.path.exists(p_path_o)):
            print('The origin zoom is not existed!')
            return False

    if zoom_in is not None:
        if not isinstance(zoom_in, list):
            zoom_in = [zoom_in]

        for i_zoom in zoom_in:
            if (zoom_origin < i_zoom):
                print('The origin zoom is less than zoom_in !')
                next
            i_zoom_str = "%.0f%%" % (i_zoom * 100)
            for p in prefix:
                p_path_o = os.path.join(main_path, p + '_' + zoom_origin_str)
                if os.path.exists(p_path_o):
                    p_path_d = os.path.join(main_path, p + '_' + i_zoom_str)
                    if not os.path.exists(p_path_d):
                        os.makedirs(p_path_d)
                    this_S_folders = os.listdir(p_path_o)
                    for this_S_folder in this_S_folders:  # this_S_folder is S1 to S96
                        this_dS_folder_path = os.path.join(p_path_d, this_S_folder)
                        if not os.path.exists(this_dS_folder_path):
                            os.makedirs(this_dS_folder_path)
                        this_images = os.listdir(os.path.join(p_path_o, this_S_folder))
                        for this_image_name in this_images:
                            this_o_image_path = os.path.join(p_path_o, this_S_folder, this_image_name)
                            d_img_path = os.path.join(this_dS_folder_path, this_image_name)

                            this_o_img = cv2.imread(this_o_image_path, -1)
                            d_img = cv2.resize(this_o_img, (0, 0), fx=i_zoom / zoom_origin, fy=i_zoom / zoom_origin,
                                               interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(d_img_path, d_img)
    else:
        i_zoom_str = str(pixel[0]) + '_' + str(pixel[1])
        for p in prefix:
            p_path_o = os.path.join(main_path, p + '_' + zoom_origin_str)
            if os.path.exists(p_path_o):
                p_path_d = os.path.join(main_path, p + '_' + i_zoom_str)
                if not os.path.exists(p_path_d):
                    os.makedirs(p_path_d)
                this_S_folders = os.listdir(p_path_o)
                for this_S_folder in this_S_folders:  # this_S_folder is S1 to S96
                    this_dS_folder_path = os.path.join(p_path_d, this_S_folder)
                    if not os.path.exists(this_dS_folder_path):
                        os.makedirs(this_dS_folder_path)
                    this_images = os.listdir(os.path.join(p_path_o, this_S_folder))
                    for this_image_name in this_images:
                        this_o_image_path = os.path.join(p_path_o, this_S_folder, this_image_name)
                        d_img_path = os.path.join(this_dS_folder_path, this_image_name)

                        this_o_img = cv2.imread(this_o_image_path, -1)
                        d_img = cv2.resize(this_o_img, tuple(pixel), interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(d_img_path, d_img)

    return True


def folder_image_resize(image_path, size=(2480, 2480)):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    for i in path_list:  # r'Label_1.png'
        img_dirfile = os.path.join(image_path, i)
        if os.path.isfile(img_dirfile):
            o_img = any_to_image(img_dirfile)
            d_img = cv2.resize(o_img, size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(img_dirfile, d_img)
        else:
            folder_image_resize(img_dirfile)

    return True


def folder_image_resize_0(image_path, size=(2480, 2480)):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    for i in path_list:  # r'Label_1.png'
        img_file = os.path.join(image_path, i)
        o_img = any_to_image(img_file)
        d_img = cv2.resize(o_img, size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(img_file, d_img)

    return True


def folder_image_cut_n_blocks(image_path, output_path, n=3):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path_list = os.listdir(image_path)
    for this_img in path_list:  # r'Label_1.png'
        img_file = os.path.join(image_path, this_img)
        o_img = any_to_image(img_file)
        height = o_img.shape[0]
        width = o_img.shape[1]
        item_width = int(width / n)
        item_height = int(height / n)

        box_list = []  # (col0, row0, col1, row1)
        for i in range(0, n):  # row
            for j in range(0, n):  # col
                box = (i * item_width, j * item_height, (i + 1) * item_width, (j + 1) * item_height)
                box_list.append(box)
        image_list = [o_img[box[1]:box[3], box[0]:box[2]] for box in box_list]

        index = 0
        for image in image_list:
            o_file = os.path.join(output_path,
                                  this_img.split('.')[0] + '_' + str(index) + '.' + this_img.split('.')[-1])
            cv2.imwrite(o_file, image)
            index += 1

    return True


def image_cut_black_margin(image_CV):
    # cut_black_margin
    # input image_CV is colored or gray image
    if len(image_CV.shape) == 2:
        tmp_img = image_CV
    elif len(image_CV.shape) == 3:
        tmp_img = np.sum(image_CV, axis=2)
    else:
        print('!Error! : wrong using cut_black_margin(): The input must be Image(np.ndarray)!')
        return None

    sum_row = np.sum(tmp_img, axis=0)
    col_start = 0
    for i in range(0, len(sum_row)):
        if sum_row[i] == 0:
            col_start = i + 1
        else:
            break
    col_end = len(sum_row)
    for i in range(len(sum_row) - 1, -1, -1):
        if sum_row[i] == 0:
            col_end = i
        else:
            break

    sum_col = np.sum(tmp_img, axis=1)
    row_start = 0
    for i in range(0, len(sum_col)):
        if sum_col[i] == 0:
            row_start = i + 1
        else:
            break
    row_end = len(sum_col)
    for i in range(len(sum_col) - 1, -1, -1):
        if sum_col[i] == 0:
            row_end = i
        else:
            break

    if len(image_CV.shape) == 2:
        result_CV = image_CV[row_start:row_end, col_start:col_end]
    elif len(image_CV.shape) == 3:
        result_CV = image_CV[row_start:row_end, col_start:col_end, :]
    else:
        result_CV = None

    return result_CV


def saving_scene_density_csv(main_path, this_index_path, T, S, avg_density):
    scene_density_mem = pd.read_csv(os.path.join(main_path, 'AVG_Density.csv'), header=0, index_col=0)
    # Scene_Density.csv 's column is::: S1-S96...
    # Scene_Density.csv 's index is::: DATE_ExperimentName_T
    t_xep_namp = os.path.split(this_index_path)
    t_date = os.path.split(t_xep_namp[0])
    this_index = t_date[1] + '~' + t_xep_namp[1] + '~T' + str(T)
    scene_density_mem.loc[this_index, 'S' + str(S)] = avg_density
    scene_density_mem.to_csv(path_or_buf=os.path.join(main_path, 'AVG_Density.csv'))
    # scene_density_mem.to_csv(path_or_buf=os.path.join(main_path, 'Scene_Density.csv'), mode='a', header=None)


def saving_density_RT(main_path, csv_file, this_index_path, T, S, density):
    # ---the 'csv_file' IS existed? or create it ---
    if not os.path.exists(os.path.join(main_path, csv_file)):
        # csv_file_mem = pd.DataFrame(columns=['S' + str(col) for col in range(1, args.S + 1)])
        csv_file_mem = pd.DataFrame()
        csv_file_mem.to_csv(path_or_buf=os.path.join(main_path, csv_file))
    else:
        csv_file_mem = pd.read_csv(os.path.join(main_path, csv_file), header=0, index_col=0)
    # 'csv_file' 's column is::: S1-S96...
    # 'csv_file' 's index is::: DATE_ExperimentName_T
    t_xep_namp = os.path.split(this_index_path)
    t_date = os.path.split(t_xep_namp[0])
    this_index = t_date[1] + '~' + t_xep_namp[1] + '~T' + str(T)
    csv_file_mem.loc[this_index, 'S' + str(S)] = density
    csv_file_mem.to_csv(path_or_buf=os.path.join(main_path, csv_file))


def saving_density(main_path, csv_file, row_index, col_index, density):
    # ---the 'csv_file' IS existed? or create it ---
    if not os.path.exists(os.path.join(main_path, csv_file)):
        # csv_file_mem = pd.DataFrame(columns=['S' + str(col) for col in range(1, args.S + 1)])
        csv_file_mem = pd.DataFrame()
        csv_file_mem.to_csv(path_or_buf=os.path.join(main_path, csv_file))
    else:
        csv_file_mem = pd.read_csv(os.path.join(main_path, csv_file), header=0, index_col=0)
    # 'csv_file' 's column is::: S1-S96...
    # 'csv_file' 's index is::: DATE_ExperimentName_T
    # t_xep_namp = os.path.split(this_index_path)
    # t_date = os.path.split(t_xep_namp[0])
    # this_index = t_date[1] + '~' + t_xep_namp[1] + '~T' + str(T)

    csv_file_mem.loc[row_index, col_index] = density
    csv_file_mem.to_csv(path_or_buf=os.path.join(main_path, csv_file))


def saving_talbe(main_path, csv_file, row_index, col_index, value):
    # ---the 'csv_file' IS existed? or create it ---
    this_file = os.path.join(main_path, csv_file)
    if not os.path.exists(this_file):
        csv_file_mem = pd.DataFrame()
        csv_file_mem.to_csv(path_or_buf=this_file)
    else:
        csv_file_mem = pd.read_csv(this_file, header=0, index_col=0)

    csv_file_mem.loc[row_index, col_index] = value
    csv_file_mem.to_csv(path_or_buf=this_file)


def saving_time(main_path, csv_file, this_index_path, T, S):
    # ---the 'csv_file' IS existed? or create it ---
    if not os.path.exists(os.path.join(main_path, csv_file)):
        # csv_file_mem = pd.DataFrame(columns=['S' + str(col) for col in range(1, args.S + 1)])
        csv_file_mem = pd.DataFrame()
        csv_file_mem.to_csv(path_or_buf=os.path.join(main_path, csv_file))
    else:
        csv_file_mem = pd.read_csv(os.path.join(main_path, csv_file), header=0, index_col=0)
    # 'csv_file' 's column is::: S1-S96...
    # 'csv_file' 's index is::: DATE_ExperimentName_T
    t_xep_namp = os.path.split(this_index_path)
    t_date = os.path.split(t_xep_namp[0])
    this_index = t_date[1] + '~' + t_xep_namp[1] + '~T' + str(T)
    csv_file_mem.loc[this_index, 'S' + str(S)] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    csv_file_mem.to_csv(path_or_buf=os.path.join(main_path, csv_file))


def square_stitching_save(main_path, path, B, T, S, Z, C, sum_M, side, zoom, overlap):
    # !!old version!!
    # can stitching 5*5=25 well image using Physics Physics plane Absolute coordinates
    # input :
    # main_path, path, B, T, S, Z, C: is the same
    # !!!B, T, S, Z, C, M!!! is from 1!!!
    # a 96well sum_M==25 side==5
    # zoom is a float list or a float, notice its prefer to be [1, 0.3] the big zoom first
    # overlap is very important with can be found in Zeiss experiment design
    # output:
    # is a 2 str tuple of zoom[0] image file path, notice only 2!
    # exp:('F:\CD13\SSS_30%\S1\IPS_CD16_M0000.jpg', 'F:\CD13\SSSS_30%\S1\IPS_CD16_M0000.jpg')
    if sum_M != 25 or side != 5:
        return None
    overlap = overlap / 2
    if side ** 2 != sum_M:
        return None
    for this_M in range(1, sum_M + 1):
        this_image_path = get_specific_image(path, B, T, S, Z, C, this_M)
        if this_image_path is not None and os.path.exists(this_image_path):
            test_image = Image.open(this_image_path)
            break
    img_width = int(test_image.width - 2 * round(overlap * test_image.width))  # width is x
    img_height = int(test_image.height - 2 * round(overlap * test_image.height))  # height is y
    box = (round(overlap * test_image.width), round(overlap * test_image.height),
           test_image.width - round(overlap * test_image.width),
           test_image.height - round(overlap * test_image.height))  # (left, upper, right, lower) (x0,y0) to (x1,y1)
    result_image = Image.new('RGB', (img_width * side, img_height * side))
    useful_result_image = Image.new('RGB', (img_width * (side - 2), img_height * (side - 2)))
    if os.path.exists(path):
        for this_M in range(1, sum_M + 1):
            this_image_path = get_specific_image(path, B, T, S, Z, C, this_M)
            if this_image_path is not None and os.path.exists(this_image_path):
                this_image = Image.open(this_image_path)
                row = (this_M - 1) // side  # row is y
                order = row % 2
                if order == 0:
                    col = (this_M - 1) % side  # col is x
                else:
                    col = side - 1 - (this_M - 1) % side
                result_image.paste(this_image.crop(box), (col * img_width, row * img_height))
                if (row > 0 or row < side - 1) and (col > 0 or col < side - 1):
                    useful_result_image.paste(this_image.crop(box), ((col - 1) * img_width, (row - 1) * img_height))

    t_xep_namp = os.path.split(path)  # D:\processing\CD16\2019-01-10    I-1_CD16
    t_date = os.path.split(t_xep_namp[0])  # D:\processing\CD16         2019-01-10

    if not isinstance(zoom, list):
        zoom = [zoom]

    result_img_path_list = []
    useful_result_img_path_list = []
    for i_zoom in zoom:
        if i_zoom == 1:
            this_result_img = result_image
            this_useful_result_img = useful_result_image
        else:
            this_result_img = result_image.resize((round(i_zoom * img_width * side), round(i_zoom * img_height * side)),
                                                  Image.ANTIALIAS)
            this_useful_result_img = useful_result_image.resize(
                (round(i_zoom * img_width * (side - 2)), round(i_zoom * img_height * (side - 2))), Image.ANTIALIAS)

        zoom_str = "%.0f%%" % (i_zoom * 100)
        SSS_path = os.path.join(main_path, 'SSS_' + zoom_str)
        SSSS_path = os.path.join(main_path, 'SSSS_' + zoom_str)

        if not os.path.exists(SSS_path):
            os.makedirs(SSS_path)
        if not os.path.exists(os.path.join(SSS_path, 'S' + str(S))):
            os.makedirs(os.path.join(SSS_path, 'S' + str(S)))

        this_path = os.path.join(SSS_path, 'S' + str(S), t_date[1] + '~' + t_xep_namp[1] + '~T' + str(T) + '.png')
        result_img_path_list.append(this_path)
        this_result_img.save(this_path)

        if not os.path.exists(SSSS_path):
            os.makedirs(SSSS_path)
        if not os.path.exists(os.path.join(SSSS_path, 'S' + str(S))):
            os.makedirs(os.path.join(SSSS_path, 'S' + str(S)))

        this_usf_path = os.path.join(SSSS_path, 'S' + str(S), t_date[1] + '~' + t_xep_namp[1] + '~T' + str(T) + '.png')
        useful_result_img_path_list.append(this_usf_path)
        this_useful_result_img.save(this_usf_path)

    return (result_img_path_list[0], useful_result_img_path_list[0])


def cut_dir_center(input_path, output_path, gray=False):
    img_list = os.listdir(input_path)
    for img in img_list:
        image = any_to_image(os.path.join(input_path, img))
        if gray:
            image = image_to_gray(image)
        out = cut_center(image)
        cv2.imwrite(os.path.join(output_path, img), out)


def cut_center(img, xblocks=5, yblocks=5, xcenter=3, ycenter=3):
    image = any_to_image(img)
    xedge = (xblocks - xcenter) / 2
    yedge = (yblocks - ycenter) / 2
    i_0 = int(image.shape[0] * yedge / yblocks)
    i_1 = int(image.shape[1] * xedge / xblocks)
    out = image[i_0:image.shape[0] - i_0, i_1:image.shape[1] - i_1]
    return out


def stitching_CZI_AE_old(main_path, path, B, T, S, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True):
    # !!old version!!
    # accoding to matrix stitching the carl zeiss auto exported images
    # input :
    # main_path is main path
    # path is D:\processing\CD16\2019-01-10\I-1_CD16
    # B, T, S, Z, C is the specific well image path
    # !!!B, T, S, Z, C, M!!! is from 1!!!
    # matrix is the CZI well fill matrix 5*5 stitching images exp:
    # matrix = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    # zoom is a float list or a float, notice its prefer to be [1, 0.3] the big zoom first
    # overlap is very important with can be found in Zeiss experiment design
    # output:
    # is a 2 str tuple of zoom[0] image file path, notice only 2!
    # exp:('F:\CD13\SSS_30%\S1\IPS_CD16_M0000.jpg', 'F:\CD13\SSSS_30%\S1\IPS_CD16_M0000.jpg')
    # SSS: Sequential Stitching Scene
    # SSSS: Square Sequential Stitching Scene (remove margin which in the 'dish_margin.txt' file under the main_path)
    # 'dish_margin.txt' Examples : 1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    if do_SSSS:
        tiles_margin = scan_dish_margin(main_path)

    if output is None:
        output = main_path
    else:
        if not os.path.exists(output):
            print('!ERROR! The output does not existed!')
            return False

    overlap = overlap / 2

    matrix = matrix_list[S - 1]
    sum_M = np.sum(matrix)
    side_h = matrix.shape[0]
    side_w = matrix.shape[1]
    loc_list = []
    for row in range(side_h):
        for col in range(side_w):
            if row % 2 != 0:
                col = side_w - 1 - col  # 4-0=4 4-1=3 4-2=2 4-3=1 4-4=0
            if (matrix[row, col] == 1):
                loc_list.append((row, col))

    for this_M in range(1, sum_M + 1):
        this_image_path = get_specific_image(path, B, T, S, Z, C, this_M)
        if this_image_path is not None and os.path.exists(this_image_path):
            test_image = Image.open(this_image_path)
            break
    img_width = int(test_image.width - 2 * round(overlap * test_image.width))  # width is x
    img_height = int(test_image.height - 2 * round(overlap * test_image.height))  # height is y

    # now using plt.Image format
    box = (round(overlap * test_image.width), round(overlap * test_image.height),
           test_image.width - round(overlap * test_image.width),
           test_image.height - round(overlap * test_image.height))  # (left, upper, right, lower) (x0,y0) to (x1,y1)
    SSS_image = Image.new('RGB', (img_width * side_w, img_height * side_h))
    if do_SSSS:
        SSSS_image = Image.new('RGB', (img_width * side_w, img_height * side_h))
    # SSSS_image = Image.new('RGB', (img_width * (side_w - 2), img_height * (side_h - 2)))

    for this_M in range(1, sum_M + 1):
        this_image_path = get_specific_image(path, B, T, S, Z, C, this_M)
        if this_image_path is not None and os.path.exists(this_image_path):
            this_image = Image.open(this_image_path)
            SSS_image.paste(this_image.crop(box),
                            (loc_list[this_M - 1][1] * img_width, loc_list[this_M - 1][0] * img_height))
            if do_SSSS:
                if this_M not in tiles_margin:
                    SSSS_image.paste(this_image.crop(box),
                                     (loc_list[this_M - 1][1] * img_width, loc_list[this_M - 1][0] * img_height))

    # now using np.ndarray image format
    SSS_image = np.asarray(SSS_image)
    # SSSS_image = Image.fromarray(cut_black_margin(np.asarray(SSSS_image)))
    if do_SSSS:
        SSSS_image = image_cut_black_margin(np.asarray(SSSS_image))

    t_xep_namp = os.path.split(path)  # D:\processing\CD16\2019-01-10    I-1_CD16
    t_date = os.path.split(t_xep_namp[0])  # D:\processing\CD16         2019-01-10

    if not isinstance(zoom, list):
        zoom = [zoom]

    SSS_list = []
    if do_SSSS:
        SSSS_list = []
    for i_zoom in zoom:
        if i_zoom == 1:
            this_SSS_img = SSS_image
            if do_SSSS:
                this_SSSS_img = SSSS_image
        else:
            # this_SSS_img = SSS_image.resize((round(i_zoom * SSS_image.width), round(i_zoom * SSS_image.height)),
            #                                 Image.ANTIALIAS)
            # this_SSSS_img = SSSS_image.resize(
            #     (round(i_zoom * SSSS_image.width), round(i_zoom * SSSS_image.width)), Image.ANTIALIAS)
            this_SSS_img = cv2.resize(SSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
            if do_SSSS:
                this_SSSS_img = cv2.resize(SSSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)

        zoom_str = "%.0f%%" % (i_zoom * 100)

        SSS_path = os.path.join(output, 'SSS_' + zoom_str)
        if not os.path.exists(SSS_path):
            os.makedirs(SSS_path)
        if not os.path.exists(os.path.join(SSS_path, 'S' + str(S))):
            os.makedirs(os.path.join(SSS_path, 'S' + str(S)))
        SSSimg_path = os.path.join(SSS_path, 'S' + str(S), t_date[1] + '~' + t_xep_namp[1] + '~T' + str(T) + '.png')
        SSS_list.append(SSSimg_path)
        # this_SSS_img.save(SSSimg_path)
        cv2.imwrite(SSSimg_path, this_SSS_img)

        if do_SSSS:
            SSSS_path = os.path.join(output, 'SSSS_' + zoom_str)
            if not os.path.exists(SSSS_path):
                os.makedirs(SSSS_path)
            if not os.path.exists(os.path.join(SSSS_path, 'S' + str(S))):
                os.makedirs(os.path.join(SSSS_path, 'S' + str(S)))

            SSSSimg_path = os.path.join(SSSS_path, 'S' + str(S),
                                        t_date[1] + '~' + t_xep_namp[1] + '~T' + str(T) + '.png')
            SSSS_list.append(SSSSimg_path)
            # this_SSSS_img.save(SSSSimg_path)
            cv2.imwrite(SSSSimg_path, this_SSSS_img)

    if do_SSSS:
        result = (SSS_list[0], SSSS_list[0])
    else:
        result = SSS_list[0]
    return result


def stitching_well_all_time(main_path, path, row, col, w=320, h=320, zoom=None, output='All_Wells', sort_function=None):
    # put all 24 or 96 wells together
    # if zoom is None, using w and h
    # path: input path exp: r'J:\CD42\PROCESSING\SSS_100%'
    # output = os.path.join(main_path, output)

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    output = os.path.join(main_path, output)
    if not os.path.exists(output):
        os.makedirs(output)

    S_list = os.listdir(path)
    if len(S_list) != row * col:
        print('!ERROR! Well number error!')
        return False

    loc_list = []
    for r in range(row):
        for c in range(col):
            if r % 2 != 0:
                c = col - 1 - c  # 4-0=4 4-1=3 4-2=2 4-3=1 4-4=0
            loc_list.append((r, c))

    img_name_list = None
    for i_S in S_list:
        i_S = os.path.join(path, i_S)
        if img_name_list is None:
            img_name_list = os.listdir(i_S)
        else:
            img_name_list = list(set(img_name_list).union(set(os.listdir(i_S))))

    if sort_function is not None:
        sort_function(img_name_list)

    if zoom is None:
        whole_img_shape = (h * row, w * col, 3)  # .shap(h,w [,color])
    else:
        w_h_list = np.zeros((row, col, 2), dtype=np.int)
        i_name = img_name_list[0]  # the first one must be full [exp: ips.png]
        for i_S in S_list:
            S_num = int(i_S.split('S')[1])
            i_img_path = os.path.join(path, i_S, i_name)
            this_image = cv2.imread(i_img_path, 1)  # no try, when error jump
            w_h_list[loc_list[S_num - 1]] = [int(zoom * this_image.shape[0]), int(zoom * this_image.shape[1])]
        # whole_img_shape = (np.sum(w_h_list, axis=(0, 1))[0], np.sum(w_h_list, axis=(0, 1))[1], 3)
        whole_img_shape = (np.sum(w_h_list[:, 0, 0]), np.sum(w_h_list[0, :, 1]), 3)

    for i_name in img_name_list:

        print('>>>Donging : stitching_well(' + i_name + ')')
        this_whole_img = np.zeros(whole_img_shape, dtype=np.uint8)
        this_img_path = os.path.join(output, i_name)

        for i_S in S_list:
            S_num = int(i_S.split('S')[1])
            i_img_path = os.path.join(path, i_S, i_name)
            if not os.path.exists(i_img_path):
                continue
            try:
                this_image = cv2.imread(i_img_path, 1)
                if zoom is None:
                    this_image = cv2.resize(this_image, (w, h), interpolation=cv2.INTER_NEAREST)
                    this_whole_img[loc_list[S_num - 1][0] * h: loc_list[S_num - 1][0] * h + h,
                    loc_list[S_num - 1][1] * w: loc_list[S_num - 1][1] * w + w] = this_image
                else:

                    # this_image = cv2.resize(this_image, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
                    # this_whole_img[loc_list[S_num - 1][0] * h: loc_list[S_num - 1][0] * h + h, loc_list[S_num - 1][1] * w: loc_list[S_num - 1][1] * w + w] = this_image
                    y0 = sum(w_h_list[0:loc_list[S_num - 1][0], 0, 0])
                    x0 = sum(w_h_list[0, 0:loc_list[S_num - 1][1], 1])
                    this_h = w_h_list[loc_list[S_num - 1][0], loc_list[S_num - 1][1], 0]
                    this_w = w_h_list[loc_list[S_num - 1][0], loc_list[S_num - 1][1], 1]
                    y1 = y0 + this_h
                    x1 = x0 + this_w
                    this_image = cv2.resize(this_image, (this_w, this_h), interpolation=cv2.INTER_NEAREST)
                    this_whole_img[y0:y1, x0:x1] = this_image

            except BaseException as e:
                print('!ERROR! ', e)
                print('!ERROR! The Imgae Error, maybe open, maybe structure broken...')
            else:
                pass
            finally:
                pass

        cv2.imwrite(this_img_path, this_whole_img)

    return


def stitching_well_one_picture(main_path, path, output, row, col, stage_str, T, w=320, h=320, zoom=None,
                               sort_function=None):
    # put all 24 or 96 wells together
    # if zoom is None, using w and h
    # path: input path exp: r'J:\CD42\PROCESSING\SSS_100%'
    # output = os.path.join(main_path, output)

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    output = os.path.join(main_path, output)
    if not os.path.exists(output):
        os.makedirs(output)

    S_list = os.listdir(path)
    if len(S_list) != row * col:
        print('!ERROR! Well number error!')
        return False

    loc_list = []
    for r in range(row):
        for c in range(col):
            if r % 2 != 0:
                c = col - 1 - c  # 4-0=4 4-1=3 4-2=2 4-3=1 4-4=0
            loc_list.append((r, c))

    img_name_list = None
    for i_S in S_list:
        i_S = os.path.join(path, i_S)
        if img_name_list is None:
            img_name_list = os.listdir(i_S)
        else:
            img_name_list = list(set(img_name_list).union(set(os.listdir(i_S))))

    if sort_function is not None:
        sort_function(img_name_list)

    if zoom is None:
        whole_img_shape = (h * row, w * col, 3)  # .shap(h,w [,color])
    else:
        w_h_list = np.zeros((row, col, 2), dtype=np.int)
        i_name = img_name_list[0]  # the first one must be full [exp: ips.png]
        for i_S in S_list:
            S_num = int(i_S.split('S')[1])
            i_img_path = os.path.join(path, i_S, i_name)
            this_image = cv2.imread(i_img_path, 1)  # no try, when error jump
            w_h_list[loc_list[S_num - 1]] = [int(zoom * this_image.shape[0]), int(zoom * this_image.shape[1])]
        # whole_img_shape = (np.sum(w_h_list, axis=(0, 1))[0], np.sum(w_h_list, axis=(0, 1))[1], 3)
        whole_img_shape = (np.sum(w_h_list[:, 0, 0]), np.sum(w_h_list[0, :, 1]), 3)

    T_str = '~T' + str(T)
    for i_name in img_name_list:
        # r'2018-11-29~IPS-2_CD13~T13.jpg'
        if i_name.find(stage_str) != -1 and i_name.find(T_str) != -1:  # find the image
            print('>>>found the ' + i_name + ' image!')
            this_whole_img = np.zeros(whole_img_shape, dtype=np.uint8)
            this_img_path = os.path.join(output, i_name)

            for i_S in S_list:
                S_num = int(i_S.split('S')[1])
                i_img_path = os.path.join(path, i_S, i_name)
                if not os.path.exists(i_img_path):
                    continue
                try:
                    this_image = cv2.imread(i_img_path, 1)
                    if zoom is None:
                        this_image = cv2.resize(this_image, (w, h), interpolation=cv2.INTER_NEAREST)
                        this_whole_img[loc_list[S_num - 1][0] * h: loc_list[S_num - 1][0] * h + h,
                        loc_list[S_num - 1][1] * w: loc_list[S_num - 1][1] * w + w] = this_image
                    else:

                        # this_image = cv2.resize(this_image, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
                        # this_whole_img[loc_list[S_num - 1][0] * h: loc_list[S_num - 1][0] * h + h, loc_list[S_num - 1][1] * w: loc_list[S_num - 1][1] * w + w] = this_image
                        y0 = sum(w_h_list[0:loc_list[S_num - 1][0], 0, 0])
                        x0 = sum(w_h_list[0, 0:loc_list[S_num - 1][1], 1])
                        this_h = w_h_list[loc_list[S_num - 1][0], loc_list[S_num - 1][1], 0]
                        this_w = w_h_list[loc_list[S_num - 1][0], loc_list[S_num - 1][1], 1]
                        y1 = y0 + this_h
                        x1 = x0 + this_w
                        this_image = cv2.resize(this_image, (this_w, this_h), interpolation=cv2.INTER_NEAREST)
                        this_whole_img[y0:y1, x0:x1] = this_image

                except BaseException as e:
                    print('!ERROR! ', e)
                    print('!ERROR! The Imgae Error, maybe open, maybe structure broken...')
                else:
                    pass
                finally:
                    pass

            cv2.imwrite(this_img_path, this_whole_img)

    return


def stitching_well_by_name(main_path, path, output, row, col, img_name, w=320, h=320, zoom=None,
                           sort_function=None):
    # put all 24 or 96 wells together
    # if zoom is None, using w and h
    # path: input path exp: r'J:\CD42\PROCESSING\SSS_100%'
    # output = os.path.join(main_path, output)

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    output = os.path.join(main_path, output)
    if not os.path.exists(output):
        os.makedirs(output)

    S_list = os.listdir(path)
    if len(S_list) != row * col:
        print('!ERROR! Well number error!')
        return False

    loc_list = []
    for r in range(row):
        for c in range(col):
            if r % 2 != 0:
                c = col - 1 - c  # 4-0=4 4-1=3 4-2=2 4-3=1 4-4=0
            loc_list.append((r, c))

    img_name_list = None
    for i_S in S_list:
        i_S = os.path.join(path, i_S)
        if img_name_list is None:
            img_name_list = os.listdir(i_S)
        else:
            img_name_list = list(set(img_name_list).union(set(os.listdir(i_S))))

    if sort_function is not None:
        sort_function(img_name_list)

    if zoom is None:
        whole_img_shape = (h * row, w * col, 3)  # .shap(h,w [,color])
    else:
        w_h_list = np.zeros((row, col, 2), dtype=np.int)
        i_name = img_name_list[0]  # the first one must be full [exp: ips.png]
        for i_S in S_list:
            S_num = int(i_S.split('S')[1])
            i_img_path = os.path.join(path, i_S, i_name)
            this_image = cv2.imread(i_img_path, 1)  # no try, when error jump
            w_h_list[loc_list[S_num - 1]] = [int(zoom * this_image.shape[0]), int(zoom * this_image.shape[1])]
        # whole_img_shape = (np.sum(w_h_list, axis=(0, 1))[0], np.sum(w_h_list, axis=(0, 1))[1], 3)
        whole_img_shape = (np.sum(w_h_list[:, 0, 0]), np.sum(w_h_list[0, :, 1]), 3)

    for i_name in img_name_list:
        # r'2018-11-29~IPS-2_CD13~T13.jpg'
        if i_name == img_name:  # find the image by name
            print('>>>found the ' + i_name + ' image!')
            this_whole_img = np.zeros(whole_img_shape, dtype=np.uint8)
            this_img_path = os.path.join(output, i_name)

            for i_S in S_list:
                S_num = int(i_S.split('S')[1])
                i_img_path = os.path.join(path, i_S, i_name)
                if not os.path.exists(i_img_path):
                    continue
                try:
                    this_image = cv2.imread(i_img_path, 1)
                    if zoom is None:
                        this_image = cv2.resize(this_image, (w, h), interpolation=cv2.INTER_NEAREST)
                        this_whole_img[loc_list[S_num - 1][0] * h: loc_list[S_num - 1][0] * h + h,
                        loc_list[S_num - 1][1] * w: loc_list[S_num - 1][1] * w + w] = this_image
                    else:

                        # this_image = cv2.resize(this_image, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
                        # this_whole_img[loc_list[S_num - 1][0] * h: loc_list[S_num - 1][0] * h + h, loc_list[S_num - 1][1] * w: loc_list[S_num - 1][1] * w + w] = this_image
                        y0 = sum(w_h_list[0:loc_list[S_num - 1][0], 0, 0])
                        x0 = sum(w_h_list[0, 0:loc_list[S_num - 1][1], 1])
                        this_h = w_h_list[loc_list[S_num - 1][0], loc_list[S_num - 1][1], 0]
                        this_w = w_h_list[loc_list[S_num - 1][0], loc_list[S_num - 1][1], 1]
                        y1 = y0 + this_h
                        x1 = x0 + this_w
                        this_image = cv2.resize(this_image, (this_w, this_h), interpolation=cv2.INTER_NEAREST)
                        this_whole_img[y0:y1, x0:x1] = this_image

                except BaseException as e:
                    print('!ERROR! ', e)
                    print('!ERROR! The Imgae Error, maybe open, maybe structure broken...')
                else:
                    pass
                finally:
                    pass

            cv2.imwrite(this_img_path, this_whole_img)

    return


def stitching_CZI(main_path, path, B, T, S, Z, C, matrix_list, zoom, overlap, output=None, suffix='', do_SSSS=True,
                  name_B=False, name_T=True, name_S=False, name_Z=False, name_C=False, do_enhancement=False):
    # accoding to matrix stitching images-export function after the Experiment finished or auto exported images
    # stitching_CZI is  :::  stitching_CZI_AE & stitching_CZI_IEXP
    # input :
    # !!!B, T, S, Z, C, M!!! is from 1!!!
    # main_path is main path which contain dish_margin.txt
    # path is D:\processing\CD16\2019-01-10\I-1_CD16
    # file is B=0\T=0\S=0\Z=2\C=0\IPS_CD13_S0000(A1-A1)_T000000_Z0000_C00_M0000_ORG.jpg
    # path is D:\processing\CD16\image_exported\2018-09-03~I-1_CD09
    # file is 2018-09-03~I-1_CD09~s01t01z1m001.png
    # B, T, S, Z, C is the specific well image path
    # matrix is the CZI well fill matrix 5*5 stitching images exp:
    # matrix = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    # matrix_list is the matrix * (well number exp:96) notice! the edge well always has different matrix
    # zoom is a float list or a float, notice its prefer to be [1, 0.3] the big zoom first
    # overlap is very important with can be found in Zeiss experiment design
    # output:
    # is a 2 str tuple of zoom[0] image file path, notice only 2!
    # exp:('F:\CD13\SSS_30%\S1\IPS_CD16_M0000.jpg', 'F:\CD13\SSSS_30%\S1\IPS_CD16_M0000.jpg')
    # SSS: Sequential Stitching Scene
    # SSSS: Square Sequential Stitching Scene (remove margin which in the 'dish_margin.txt' file under the main_path)
    # 'dish_margin.txt' Examples : 1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    if do_SSSS:
        tiles_margin = scan_dish_margin(main_path)

    if output is None:
        output = main_path
    else:
        if not os.path.exists(output):
            print('!ERROR! The output does not existed!')
            return False

    overlap = overlap / 2

    matrix = matrix_list[S - 1]
    sum_M = np.sum(matrix)
    side_h = matrix.shape[0]
    side_w = matrix.shape[1]
    loc_list = []
    for row in range(side_h):
        for col in range(side_w):
            if row % 2 != 0:
                col = side_w - 1 - col  # 4-0=4 4-1=3 4-2=2 4-3=1 4-4=0
            if (matrix[row, col] == 1):
                loc_list.append((row, col))

    for this_M in range(1, sum_M + 1):
        img_pth_list = get_CZI_image(path, B, T, S, Z, C, this_M)
        if img_pth_list is not None and os.path.exists(img_pth_list[0]):
            # test_image = Image.open(img_pth_list[0])
            test_image = cv2.imread(img_pth_list[0], -1)  # .shap(h,w [,color])
            test_image_height = test_image.shape[0]
            test_image_width = test_image.shape[1]
            break
    img_width = int(test_image_width - 2 * round(overlap * test_image_width))  # width is x
    img_height = int(test_image_height - 2 * round(overlap * test_image_height))  # height is y
    img_lift_outer_margin = round(overlap * test_image_width)
    img_top_outer_margin = round(overlap * test_image_height)

    # now using plt.Image format
    # box = (round(overlap * test_image.width), round(overlap * test_image.height),
    #        test_image.width - round(overlap * test_image.width),
    #        test_image.height - round(overlap * test_image.height))  # (left, upper, right, lower) (x0,y0) to (x1,y1)

    # SSS_image = Image.new('RGB', (img_width * side_w, img_height * side_h))
    if len(test_image.shape) == 2:
        whole_img_shape = (img_height * side_h, img_width * side_w)  # .shap(h,w [,color])
    else:
        whole_img_shape = (img_height * side_h, img_width * side_w, 3)  # .shap(h,w [,color])

    if do_enhancement:
        SSS_image = np.zeros(whole_img_shape[:2], dtype=np.uint8)
        if do_SSSS:
            SSSS_image = np.zeros(whole_img_shape[:2], dtype=np.uint8)
    else:
        SSS_image = np.zeros(whole_img_shape, dtype=test_image.dtype)
        if do_SSSS:
            SSSS_image = np.zeros(whole_img_shape, dtype=test_image.dtype)
            # SSSS_image = Image.new('RGB', (img_width * side_w, img_height * side_h))
            # SSSS_image = Image.new('RGB', (img_width * (side_w - 2), img_height * (side_h - 2)))

    date_name = None
    exp_name = None

    for this_M in range(1, sum_M + 1):
        img_pth_list = get_CZI_image(path, B, T, S, Z, C, this_M)
        if img_pth_list is not None and os.path.exists(img_pth_list[0]):
            if date_name is None or exp_name is None:
                # img_pth_list=[img_path,'S1','2018-09-03','I-1_CD09','T1','Z1']
                date_name = img_pth_list[2]
                exp_name = img_pth_list[3]
            # this_image = Image.open(img_pth_list[0])
            try:
                # print(img_pth_list[0])
                this_image = cv2.imread(img_pth_list[0], -1)
                if do_enhancement:
                    this_image = image_retreatment_toGray(img_pth_list[0])

                SSS_image[loc_list[this_M - 1][0] * img_height: loc_list[this_M - 1][0] * img_height + img_height,
                loc_list[this_M - 1][1] * img_width: loc_list[this_M - 1][1] * img_width + img_width] = this_image[
                                                                                                        img_top_outer_margin: img_top_outer_margin + img_height,
                                                                                                        img_lift_outer_margin: img_lift_outer_margin + img_width]
                # loc_list is tuple(row, col) list
                # loc_list[this_M - 1][0] * img_height : loc_list[this_M - 1][0] * img_height + img_height
                # loc_list[this_M - 1][1] * img_width : loc_list[this_M - 1][1] * img_width + img_width
                # img_top_outer_margin : img_top_outer_margin + img_height
                # img_lift_outer_margin : img_lift_outer_margin + img_width

                # SSS_image.paste(this_image.crop(box),
                #                 (loc_list[this_M - 1][1] * img_width, loc_list[this_M - 1][0] * img_height))
                if do_SSSS:
                    if this_M not in tiles_margin:
                        # SSSS_image.paste(this_image.crop(box),
                        #                  (loc_list[this_M - 1][1] * img_width, loc_list[this_M - 1][0] * img_height))
                        SSSS_image[
                        loc_list[this_M - 1][0] * img_height: loc_list[this_M - 1][0] * img_height + img_height,
                        loc_list[this_M - 1][1] * img_width: loc_list[this_M - 1][
                                                                 1] * img_width + img_width] = this_image[
                                                                                               img_top_outer_margin: img_top_outer_margin + img_height,
                                                                                               img_lift_outer_margin: img_lift_outer_margin + img_width]
            except BaseException as e:
                print('!ERROR! ', e)
                print('!ERROR! The Imgae Error, maybe open, maybe structure broken...')
                print('!ERROR! Imgae:', path, 'B=', B, 'T=', T, 'S=', S, 'z=', Z, 'C=', C, 'M=', this_M)
            else:
                pass
            finally:
                pass

    # now using np.ndarray image format
    # SSS_image = np.asarray(SSS_image)
    # SSSS_image = Image.fromarray(cut_black_margin(np.asarray(SSSS_image)))
    if do_SSSS:
        # SSSS_image = image_cut_black_margin(np.asarray(SSSS_image))
        SSSS_image = image_cut_black_margin(SSSS_image)

    if not isinstance(zoom, list):
        zoom = [zoom]

    SSS_list = []
    if do_SSSS:
        SSSS_list = []
    for i_zoom in zoom:
        if i_zoom == 1:
            this_SSS_img = SSS_image
            if do_SSSS:
                this_SSSS_img = SSSS_image
        else:
            # this_SSS_img = SSS_image.resize((round(i_zoom * SSS_image.width), round(i_zoom * SSS_image.height)),
            #                                 Image.ANTIALIAS)
            # this_SSSS_img = SSSS_image.resize(
            #     (round(i_zoom * SSSS_image.width), round(i_zoom * SSSS_image.width)), Image.ANTIALIAS)
            # this_SSS_img = cv2.resize(SSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
            try:
                this_SSS_img = cv2.resize(SSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
            except BaseException as e:
                print('!ERROR! ', e)
                this_SSS_img = np.zeros((img_height, img_width), dtype=test_image.dtype)
            else:
                pass
            finally:
                pass
            if do_SSSS:
                # this_SSSS_img = cv2.resize(SSSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
                try:
                    this_SSSS_img = cv2.resize(SSSS_image, (0, 0), fx=i_zoom, fy=i_zoom,
                                               interpolation=cv2.INTER_NEAREST)
                except BaseException as e:
                    print('!ERROR! ', e)
                    this_SSSS_img = np.zeros((img_height, img_width), dtype=test_image.dtype)
                else:
                    pass
                finally:
                    pass

        zoom_str = "%.0f%%" % (i_zoom * 100)

        SSS_path = os.path.join(output, 'SSS_' + zoom_str + suffix)  # r'J:\PROCESSING\CD13\SSS_100%
        if not os.path.exists(SSS_path):
            os.makedirs(SSS_path)

        SSS_S_path = os.path.join(SSS_path, 'S' + str(S))  # r'J:\PROCESSING\CD13\SSS_100%\S1
        if not os.path.exists(SSS_S_path):
            os.makedirs(SSS_S_path)

        # r'2018-11-28~IPS_CD13~B1~T1~S1~Z1~C1.png'
        img_file_name = date_name + '~' + exp_name

        if name_B:
            img_file_name = img_file_name + '~B' + str(B)
        if name_T:
            img_file_name = img_file_name + '~T' + str(T)
        if name_S:
            img_file_name = img_file_name + '~S' + str(S)
        if name_Z:
            img_file_name = img_file_name + '~Z' + str(Z)
        if name_C:
            img_file_name = img_file_name + '~C' + str(C)

        img_file_name = img_file_name + '.png'

        SSS_img_path = os.path.join(SSS_S_path, img_file_name)

        SSS_list.append(SSS_img_path)
        # this_SSS_img.save(SSSimg_path)
        cv2.imwrite(SSS_img_path, this_SSS_img)

        if do_SSSS:

            SSSS_path = os.path.join(output, 'SSSS_' + zoom_str + suffix)
            if not os.path.exists(SSSS_path):
                os.makedirs(SSSS_path)

            SSSS_S_path = os.path.join(SSSS_path, 'S' + str(S))
            if not os.path.exists(SSSS_S_path):
                os.makedirs(SSSS_S_path)

            # r'J:\PROCESSING\CD13\SSSS_100%\S1\2018-11-28~IPS_CD13~T1.png'
            SSSS_img_path = os.path.join(SSSS_S_path, img_file_name)

            SSSS_list.append(SSSS_img_path)
            # this_SSSS_img.save(SSSSimg_path)
            cv2.imwrite(SSSS_img_path, this_SSSS_img)

    if do_SSSS:
        result = (SSS_list[0], SSSS_list[0])
    else:
        result = SSS_list[0]
    return result


def stitching_CZI_AutoBestZ_old(main_path, path, B, T, S, C, matrix_list, zoom, overlap, output=None, do_SSSS=True):
    # accoding to matrix stitching images-export function after the Experiment finished or auto exported images
    # ! and auto find the best focus Z !
    # ! SSSS export is (invalid) !
    # stitching_CZI is  :::  stitching_CZI_AE & stitching_CZI_IEXP
    # input :
    # !!!B, T, S, Z, C, M!!! is from 1!!!
    # main_path is main path which contain dish_margin.txt
    # path is D:\processing\CD16\2019-01-10\I-1_CD16
    # file is B=0\T=0\S=0\Z=2\C=0\IPS_CD13_S0000(A1-A1)_T000000_Z0000_C00_M0000_ORG.jpg
    # path is D:\processing\CD16\image_exported\2018-09-03~I-1_CD09
    # file is 2018-09-03~I-1_CD09~s01t01z1m001.png
    # B, T, S, Z, C is the specific well image path
    # matrix is the CZI well fill matrix 5*5 stitching images exp:
    # matrix = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    # zoom is a float list or a float, notice its prefer to be [1, 0.3] the big zoom first
    # overlap is very important with can be found in Zeiss experiment design
    # output:
    # is a 2 str tuple of zoom[0] image file path, notice only 2!
    # exp:('F:\CD13\SSS_30%\S1\IPS_CD16_M0000.jpg', 'F:\CD13\SSSS_30%\S1\IPS_CD16_M0000.jpg')
    # SSS: Sequential Stitching Scene
    # SSSS: Square Sequential Stitching Scene (remove margin which in the 'dish_margin.txt' file under the main_path)
    # 'dish_margin.txt' Examples : 1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    if do_SSSS:
        tiles_margin = scan_dish_margin(main_path)

    if output is None:
        output = main_path
    else:
        if not os.path.exists(output):
            print('!ERROR! The output does not existed!')
            return False

    overlap = overlap / 2

    matrix = matrix_list[S - 1]
    sum_M = np.sum(matrix)
    side_h = matrix.shape[0]
    side_w = matrix.shape[1]
    loc_list = []
    for row in range(side_h):
        for col in range(side_w):
            if row % 2 != 0:
                col = side_w - 1 - col  # 4-0=4 4-1=3 4-2=2 4-3=1 4-4=0
            if (matrix[row, col] == 1):
                loc_list.append((row, col))

    demo_Z = 1
    for this_M in range(1, sum_M + 1):
        img_pth_list = get_CZI_image(path, B, T, S, demo_Z, C, this_M)
        if img_pth_list is not None and os.path.exists(img_pth_list[0]):
            test_image = Image.open(img_pth_list[0])
            break
    img_width = int(test_image.width - 2 * round(overlap * test_image.width))  # width is x
    img_height = int(test_image.height - 2 * round(overlap * test_image.height))  # height is y

    # now using plt.Image format
    # box: each M image removed the overlap margin
    box = (round(overlap * test_image.width), round(overlap * test_image.height),
           test_image.width - round(overlap * test_image.width),
           test_image.height - round(overlap * test_image.height))  # (left, upper, right, lower) (x0,y0) to (x1,y1)
    SSS_image = Image.new('RGB', (img_width * side_w, img_height * side_h))
    if do_SSSS:
        SSSS_image = Image.new('RGB', (img_width * side_w, img_height * side_h))
    # SSSS_image = Image.new('RGB', (img_width * (side_w - 2), img_height * (side_h - 2)))

    date_name = None
    exp_name = None

    for this_M in range(1, sum_M + 1):

        z = 1
        z_img = get_CZI_image(path, B, T, S, z, C, this_M)
        z_imgVar_list = []
        while z_img is not None:
            z_imgVar_list.append(get_ImageVar(z_img[0]))
            z += 1
            z_img = get_CZI_image(path, B, T, S, z, C, this_M)
        if z_imgVar_list:  # this M is existed
            Z = z_imgVar_list.index(max(z_imgVar_list)) + 1
        else:  # this M is not existed!
            Z = -1  # this Z=-1 is not existed!

        img_pth_list = get_CZI_image(path, B, T, S, Z, C, this_M)
        if img_pth_list is not None and os.path.exists(img_pth_list[0]):
            print('!NOTICE!', path, 'B=', B, 'T=', T, 'S=', S, 'C=', C, 'M=', this_M, 'is using Z=', Z, 'ImageVar=',
                  max(z_imgVar_list))
        else:
            print('!NOTICE!', path, 'B=', B, 'T=', T, 'S=', S, 'C=', C, 'M=', this_M, 'is not existed!')

        if img_pth_list is not None and os.path.exists(img_pth_list[0]):
            this_image = Image.open(img_pth_list[0])
            SSS_image.paste(this_image.crop(box),
                            (loc_list[this_M - 1][1] * img_width, loc_list[this_M - 1][0] * img_height))
            if do_SSSS:
                if this_M not in tiles_margin:
                    SSSS_image.paste(this_image.crop(box),
                                     (loc_list[this_M - 1][1] * img_width, loc_list[this_M - 1][0] * img_height))
            if date_name is None or exp_name is None:
                # img_pth_list=[img_path,'S1','2018-09-03','I-1_CD09','T1','Z1']
                date_name = img_pth_list[2]
                exp_name = img_pth_list[3]

    # now using np.ndarray image format
    SSS_image = np.asarray(SSS_image)
    # SSSS_image = Image.fromarray(cut_black_margin(np.asarray(SSSS_image)))
    if do_SSSS:
        SSSS_image = image_cut_black_margin(np.asarray(SSSS_image))

    if not isinstance(zoom, list):
        zoom = [zoom]

    SSS_list = []
    if do_SSSS:
        SSSS_list = []
    for i_zoom in zoom:
        if i_zoom == 1:
            this_SSS_img = SSS_image
            if do_SSSS:
                this_SSSS_img = SSSS_image
        else:
            # this_SSS_img = SSS_image.resize((round(i_zoom * SSS_image.width), round(i_zoom * SSS_image.height)),
            #                                 Image.ANTIALIAS)
            # this_SSSS_img = SSSS_image.resize(
            #     (round(i_zoom * SSSS_image.width), round(i_zoom * SSSS_image.width)), Image.ANTIALIAS)
            # interpolation = cv2.INTER_NEAREST)  # 最近邻插值 缩小最锐利
            # interpolation=cv2.INTER_LINEAR)  # 双线性插值（默认设置） 缩小平滑
            # interpolation=cv2.INTER_AREA)  # 使用像素区域关系进行重采样 缩小平滑
            # interpolation=cv2.INTER_CUBIC)  # 4x4像素邻域的双三次插值 缩小有杂色有方块
            # interpolation=cv2.INTER_LANCZOS4)  # 8x8像素邻域的Lanczos插值 缩小有杂色
            # this_SSS_img = cv2.resize(SSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
            try:
                this_SSS_img = cv2.resize(SSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
            except BaseException as e:
                print('!ERROR! ', e)
                this_SSS_img = np.zeros((img_height, img_width), dtype=test_image.dtype)
            else:
                pass
            finally:
                pass
            if do_SSSS:
                # this_SSSS_img = cv2.resize(SSSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
                try:
                    this_SSSS_img = cv2.resize(SSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
                except BaseException as e:
                    print('!ERROR! ', e)
                    this_SSSS_img = np.zeros((img_height, img_width), dtype=test_image.dtype)
                else:
                    pass
                finally:
                    pass

        zoom_str = "%.0f%%" % (i_zoom * 100)

        SSS_path = os.path.join(output, 'SSS_' + zoom_str)
        if not os.path.exists(SSS_path):
            os.makedirs(SSS_path)
        if not os.path.exists(os.path.join(SSS_path, 'S' + str(S))):
            os.makedirs(os.path.join(SSS_path, 'S' + str(S)))
        # r'J:\PROCESSING\CD13\SSS_100%\S1\2018-11-28~IPS_CD13~T1.png'
        SSSimg_path = os.path.join(SSS_path, 'S' + str(S), date_name + '~' + exp_name + '~T' + str(T) + '.png')
        SSS_list.append(SSSimg_path)
        # this_SSS_img.save(SSSimg_path)
        cv2.imwrite(SSSimg_path, this_SSS_img)

        if do_SSSS:
            SSSS_path = os.path.join(output, 'SSSS_' + zoom_str)
            if not os.path.exists(SSSS_path):
                os.makedirs(SSSS_path)
            if not os.path.exists(os.path.join(SSSS_path, 'S' + str(S))):
                os.makedirs(os.path.join(SSSS_path, 'S' + str(S)))
            # r'J:\PROCESSING\CD13\SSSS_100%\S1\2018-11-28~IPS_CD13~T1.png'
            SSSSimg_path = os.path.join(SSSS_path, 'S' + str(S), date_name + '~' + exp_name + '~T' + str(T) + '.png')
            SSSS_list.append(SSSSimg_path)
            # this_SSSS_img.save(SSSSimg_path)
            cv2.imwrite(SSSSimg_path, this_SSSS_img)

    if do_SSSS:
        result = (SSS_list[0], SSSS_list[0])
    else:
        result = SSS_list[0]
    return result


def stitching_CZI_AutoBestZ(main_path, path, B, T, S, C, matrix_list, zoom, overlap, output=None, suffix='',
                            do_SSSS=True, name_B=False, name_T=True, name_S=False, name_Z=False, name_C=False,
                            do_enhancement=False):
    # accoding to matrix stitching images-export function after the Experiment finished or auto exported images
    # ! and auto find the best focus Z !
    # ! SSSS export is (invalid) !
    # stitching_CZI is  :::  stitching_CZI_AE & stitching_CZI_IEXP
    # input :
    # !!!B, T, S, Z, C, M!!! is from 1!!!
    # main_path is main path which contain dish_margin.txt
    # path is D:\processing\CD16\2019-01-10\I-1_CD16
    # file is B=0\T=0\S=0\Z=2\C=0\IPS_CD13_S0000(A1-A1)_T000000_Z0000_C00_M0000_ORG.jpg
    # path is D:\processing\CD16\image_exported\2018-09-03~I-1_CD09
    # file is 2018-09-03~I-1_CD09~s01t01z1m001.png
    # B, T, S, Z, C is the specific well image path
    # matrix is the CZI well fill matrix 5*5 stitching images exp:
    # matrix = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    # zoom is a float list or a float, notice its prefer to be [1, 0.3] the big zoom first
    # overlap is very important with can be found in Zeiss experiment design
    # output:
    # is a 2 str tuple of zoom[0] image file path, notice only 2!
    # exp:('F:\CD13\SSS_30%\S1\IPS_CD16_M0000.jpg', 'F:\CD13\SSSS_30%\S1\IPS_CD16_M0000.jpg')
    # SSS: Sequential Stitching Scene
    # SSSS: Square Sequential Stitching Scene (remove margin which in the 'dish_margin.txt' file under the main_path)
    # 'dish_margin.txt' Examples : 1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    if do_SSSS:
        tiles_margin = scan_dish_margin(main_path)

    if output is None:
        output = main_path
    else:
        if not os.path.exists(output):
            print('!ERROR! The output does not existed!')
            return False

    overlap = overlap / 2

    matrix = matrix_list[S - 1]
    sum_M = np.sum(matrix)
    side_h = matrix.shape[0]
    side_w = matrix.shape[1]
    loc_list = []  # location_list is 1 Dim list; len is sum_M; has tuple(row, col)
    for row in range(side_h):
        for col in range(side_w):
            if row % 2 != 0:
                col = side_w - 1 - col  # 4-0=4 4-1=3 4-2=2 4-3=1 4-4=0
            if (matrix[row, col] == 1):
                loc_list.append((row, col))

    demo_Z = 1
    for this_M in range(1, sum_M + 1):
        img_pth_list = get_CZI_image(path, B, T, S, demo_Z, C, this_M)
        if img_pth_list is not None and os.path.exists(img_pth_list[0]):
            # test_image = Image.open(img_pth_list[0])
            test_image = cv2.imread(img_pth_list[0], -1)  # .shap(h,w [,color])
            test_image_height = test_image.shape[0]
            test_image_width = test_image.shape[1]
            break
    img_width = int(test_image_width - 2 * round(overlap * test_image_width))  # width is x is col
    img_height = int(test_image_height - 2 * round(overlap * test_image_height))  # height is y is row
    img_lift_outer_margin = round(overlap * test_image_width)
    img_top_outer_margin = round(overlap * test_image_height)

    # now using plt.Image format
    # box: each M image removed the overlap margin
    # box = (round(overlap * test_image_width), round(overlap * test_image_height),
    #        test_image_width - round(overlap * test_image_width),
    #        test_image_height - round(overlap * test_image_height))
    #  (left, upper, right, lower) (x0,y0) to (x1,y1)

    # SSS_image = Image.new('RGB', (img_width * side_w, img_height * side_h))
    if len(test_image.shape) == 2:
        whole_img_shape = (img_height * side_h, img_width * side_w)  # .shap(h,w [,color])
    else:
        whole_img_shape = (img_height * side_h, img_width * side_w, 3)  # .shap(h,w [,color])

    if do_enhancement:
        SSS_image = np.zeros(whole_img_shape[:2], dtype=np.uint8)
        if do_SSSS:
            SSSS_image = np.zeros(whole_img_shape[:2], dtype=np.uint8)
    else:
        SSS_image = np.zeros(whole_img_shape, dtype=test_image.dtype)
        if do_SSSS:
            SSSS_image = np.zeros(whole_img_shape, dtype=test_image.dtype)
            # SSSS_image = Image.new('RGB', (img_width * side_w, img_height * side_h))
            # SSSS_image = Image.new('RGB', (img_width * (side_w - 2), img_height * (side_h - 2)))

    date_name = None
    exp_name = None

    for this_M in range(1, sum_M + 1):

        z = 1
        z_img = get_CZI_image(path, B, T, S, z, C, this_M)
        z_imgVar_list = []
        while z_img is not None:
            z_imgVar_list.append(get_ImageVar(z_img[0]))
            z += 1
            z_img = get_CZI_image(path, B, T, S, z, C, this_M)
        if z_imgVar_list:  # this M is existed
            Z = z_imgVar_list.index(max(z_imgVar_list)) + 1
        else:  # this M is not existed!
            Z = -1  # this Z=-1 is not existed!

        if Z == -1:
            print('!ERROR! Missing', path, 'B=', B, 'T=', T, 'S=', S, 'z=', z, 'C=', C, 'M=', this_M)
            Z = 1

        img_pth_list = get_CZI_image(path, B, T, S, Z, C, this_M)
        # if img_pth_list is not None and os.path.exists(img_pth_list[0]):
        #     print('!NOTICE!', path, 'B=', B, 'T=', T, 'S=', S, 'C=', C, 'M=', this_M, 'is using Z=', Z, 'ImageVar=',
        #           max(z_imgVar_list))
        # else:
        #     print('!NOTICE!', path, 'B=', B, 'T=', T, 'S=', S, 'C=', C, 'M=', this_M, 'is not existed!')

        if img_pth_list is not None and os.path.exists(img_pth_list[0]):
            if date_name is None or exp_name is None:
                # img_pth_list=[img_path,'S1','2018-09-03','I-1_CD09','T1','Z1','C1']
                date_name = img_pth_list[2]
                exp_name = img_pth_list[3]
            # this_image = Image.open(img_pth_list[0])
            # print('!NOTICE! The mosaic tile is :::', img_pth_list[0])
            try:
                # print(img_pth_list[0])
                this_image = cv2.imread(img_pth_list[0], -1)
                if do_enhancement:
                    this_image = image_retreatment_toGray(img_pth_list[0])
                # print('this_M=', this_M, this_image.shape)
                # print('!NOTICE! The  tile shape is :::', this_image.shape)
                # print('!NOTICE! The  img_height is :::', img_height)
                # print('!NOTICE! The   img_width is :::', img_width)
                # print('!NOTICE! The  top_margin is :::', img_top_outer_margin)
                # print('!NOTICE! The left_margin is :::', img_lift_outer_margin)
                #
                # print(loc_list[this_M - 1][0] * img_height)
                # print(loc_list[this_M - 1][0] * img_height + img_height)
                # print(loc_list[this_M - 1][1] * img_width)
                # print(loc_list[this_M - 1][1] * img_width + img_width)
                #
                # print(img_top_outer_margin)
                # print(img_top_outer_margin + img_height)
                # print(img_lift_outer_margin)
                # print(img_lift_outer_margin + img_width)

                SSS_image[loc_list[this_M - 1][0] * img_height: loc_list[this_M - 1][0] * img_height + img_height,
                loc_list[this_M - 1][1] * img_width: loc_list[this_M - 1][1] * img_width + img_width] = this_image[
                                                                                                        img_top_outer_margin: img_top_outer_margin + img_height,
                                                                                                        img_lift_outer_margin: img_lift_outer_margin + img_width]

                # loc_list is tuple(row, col) list
                # loc_list[this_M - 1][0] * img_height : loc_list[this_M - 1][0] * img_height + img_height
                # loc_list[this_M - 1][1] * img_width : loc_list[this_M - 1][1] * img_width + img_width
                # img_top_outer_margin : img_top_outer_margin + img_height
                # img_lift_outer_margin : img_lift_outer_margin + img_width

                # SSS_image.paste(this_image.crop(box),
                #                 (loc_list[this_M - 1][1] * img_width, loc_list[this_M - 1][0] * img_height))
                if do_SSSS:
                    if this_M not in tiles_margin:
                        # SSSS_image.paste(this_image.crop(box),
                        #                  (loc_list[this_M - 1][1] * img_width, loc_list[this_M - 1][0] * img_height))

                        SSSS_image[
                        loc_list[this_M - 1][0] * img_height: loc_list[this_M - 1][0] * img_height + img_height,
                        loc_list[this_M - 1][1] * img_width: loc_list[this_M - 1][
                                                                 1] * img_width + img_width] = this_image[
                                                                                               img_top_outer_margin: img_top_outer_margin + img_height,
                                                                                               img_lift_outer_margin: img_lift_outer_margin + img_width]
            except BaseException as e:
                print('!ERROR! ', e)
                print('!ERROR! The Imgae Error, maybe open, maybe structure broken...')
                print('!ERROR! Imgae:', path, 'B=', B, 'T=', T, 'S=', S, 'z=', z, 'C=', C, 'M=', this_M)
            else:
                pass
            finally:
                pass

    # now using np.ndarray image format
    # SSS_image = np.asarray(SSS_image)
    # SSSS_image = Image.fromarray(cut_black_margin(np.asarray(SSSS_image)))
    if do_SSSS:
        # SSSS_image = image_cut_black_margin(np.asarray(SSSS_image))
        SSSS_image = image_cut_black_margin(SSSS_image)

    if not isinstance(zoom, list):
        zoom = [zoom]

    SSS_list = []
    if do_SSSS:
        SSSS_list = []
    for i_zoom in zoom:
        if i_zoom == 1:
            this_SSS_img = SSS_image
            if do_SSSS:
                this_SSSS_img = SSSS_image
        else:
            # this_SSS_img = SSS_image.resize((round(i_zoom * SSS_image.width), round(i_zoom * SSS_image.height)),
            #                                 Image.ANTIALIAS)
            # this_SSSS_img = SSSS_image.resize(
            #     (round(i_zoom * SSSS_image.width), round(i_zoom * SSSS_image.width)), Image.ANTIALIAS)
            # interpolation = cv2.INTER_NEAREST)  # 最近邻插值 缩小最锐利
            # interpolation=cv2.INTER_LINEAR)  # 双线性插值（默认设置） 缩小平滑
            # interpolation=cv2.INTER_AREA)  # 使用像素区域关系进行重采样 缩小平滑
            # interpolation=cv2.INTER_CUBIC)  # 4x4像素邻域的双三次插值 缩小有杂色有方块
            # interpolation=cv2.INTER_LANCZOS4)  # 8x8像素邻域的Lanczos插值 缩小有杂色
            # this_SSS_img = cv2.resize(SSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
            try:
                this_SSS_img = cv2.resize(SSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
            except BaseException as e:
                print('!ERROR! ', e)
                this_SSS_img = np.zeros((img_height, img_width), dtype=test_image.dtype)
            else:
                pass
            finally:
                pass
            if do_SSSS:
                # this_SSSS_img = cv2.resize(SSSS_image, (0, 0), fx=i_zoom, fy=i_zoom, interpolation=cv2.INTER_NEAREST)
                try:
                    this_SSSS_img = cv2.resize(SSSS_image, (0, 0), fx=i_zoom, fy=i_zoom,
                                               interpolation=cv2.INTER_NEAREST)
                except BaseException as e:
                    print('!ERROR! ', e)
                    this_SSSS_img = np.zeros((img_height, img_width), dtype=test_image.dtype)
                else:
                    pass
                finally:
                    pass

        zoom_str = "%.0f%%" % (i_zoom * 100)

        SSS_path = os.path.join(output, 'SSS_' + zoom_str + suffix)  # r'J:\PROCESSING\CD13\SSS_100%
        if not os.path.exists(SSS_path):
            os.makedirs(SSS_path)

        SSS_S_path = os.path.join(SSS_path, 'S' + str(S))  # r'J:\PROCESSING\CD13\SSS_100%\S1
        if not os.path.exists(SSS_S_path):
            os.makedirs(SSS_S_path)

        # r'2018-11-28~IPS_CD13~B1~T1~S1~Z1~C1.png'
        img_file_name = date_name + '~' + exp_name

        if name_B:
            img_file_name = img_file_name + '~B' + str(B)
        if name_T:
            img_file_name = img_file_name + '~T' + str(T)
        if name_S:
            img_file_name = img_file_name + '~S' + str(S)
        if name_Z:
            img_file_name = img_file_name + '~Zautobest'
        if name_C:
            img_file_name = img_file_name + '~C' + str(C)

        img_file_name = img_file_name + '.png'

        SSS_img_path = os.path.join(SSS_S_path, img_file_name)

        SSS_list.append(SSS_img_path)
        # this_SSS_img.save(SSSimg_path)
        cv2.imwrite(SSS_img_path, this_SSS_img)

        if do_SSSS:

            SSSS_path = os.path.join(output, 'SSSS_' + zoom_str + suffix)  # r'J:\PROCESSING\CD13\SSS_100%
            if not os.path.exists(SSSS_path):
                os.makedirs(SSSS_path)

            SSSS_S_path = os.path.join(SSSS_path, 'S' + str(S))  # r'J:\PROCESSING\CD13\SSSS_100%
            if not os.path.exists(SSSS_S_path):
                os.makedirs(SSSS_S_path)

            # r'J:\PROCESSING\CD13\SSSS_100%\S1\2018-11-28~IPS_CD13~T1.png'
            SSSS_img_path = os.path.join(SSSS_S_path, img_file_name)

            SSSS_list.append(SSSS_img_path)
            # this_SSSS_img.save(SSSSimg_path)
            cv2.imwrite(SSSS_img_path, this_SSSS_img)

    if do_SSSS:
        result = (SSS_list[0], SSSS_list[0])
    else:
        result = SSS_list[0]
    return result


def stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True,
                           name_C=False):
    # stitching carl zeiss image exported images which had already been existed on the disk
    # input
    # main_path is main path which contain dish_margin.txt
    # path is image path
    # Z, C is which Z, C wanted
    # matrix, zoom, overlap,
    # output=None

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    if output is None:
        output = main_path
    else:
        if not os.path.exists(output):
            print('!ERROR! The output does not existed!')
            return False

    # files = os.listdir(path)
    # files.sort
    # for file in files:
    #     pass
    m = 1
    s = 1
    t = 1
    while get_CZI_image(path, B, t, s, Z, C, m) is not None:
        while get_CZI_image(path, B, t, s, Z, C, m) is not None:
            stitching_CZI(main_path, path, B, t, s, Z, C, matrix_list, zoom, overlap, output=output, do_SSSS=do_SSSS,
                          name_C=name_C)
            t += 1
        else:
            print('!Warning! : Missing the T:', path, 'S=', s, 'T=', t)
        s += 1
        t = 1
    else:
        print('!Warning! : Missing the S:', path, 'S=', s)

    return True


def stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
                                     suffix='', do_SSSS=True, name_C=False, name_T=True, name_S=False,
                                     do_enhancement=False):
    # stitching carl zeiss image exported images which had already been existed on the disk
    # !and auto find the best focus Z!
    # input
    # main_path is main path which contain dish_margin.txt
    # path is image path. like: r'L:\CD33\PROCESSING\2019-08-01\CD33_STAGE2_144H_[D5end_S12]'
    # Z, C is which Z, C wanted
    # matrix, zoom, overlap,
    # output=None

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    if output is None:
        output = main_path
    else:
        if not os.path.exists(output):
            print('!ERROR! The output does not existed!')
            return False

    # files = os.listdir(path)
    # files.sort
    # for file in files:
    #     pass
    m = 1  # if one tile existed then stitching
    s = S
    t = T
    demo_z = 1
    while get_CZI_image(path, B, t, s, demo_z, C, m) is not None:  # S while
        while get_CZI_image(path, B, t, s, demo_z, C, m) is not None:  # T while
            print('stitching_CZI_AutoBestZ(t=', t, ',s=', s, ',C=', C, ',zoom=', zoom, ')')
            stitching_CZI_AutoBestZ(main_path, path, B, t, s, C, matrix_list, zoom, overlap, output=output,
                                    suffix=suffix, do_SSSS=do_SSSS, name_C=name_C, name_B=False, name_T=name_T,
                                    name_S=name_S, name_Z=False, do_enhancement=do_enhancement)

            t += 1
        else:
            print('!Warning! : Missing the T:', path, 'S=', s, 'T=', t)
        s += 1
        t = T
    else:
        print('!Warning! : Missing the S:', path, 'S=', s)

    return True


def stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None, suffix='',
                                          do_SSSS=True, name_C=True, name_T=False, name_S=False, do_enhancement=False):
    for each_C in range(1, all_C + 1):
        # print(each_C)
        stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, each_C, matrix_list, zoom, overlap, output=output,
                                         suffix=suffix, do_SSSS=do_SSSS, name_C=name_C, name_T=name_T, name_S=name_S,
                                         do_enhancement=do_enhancement)
    return True


def stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
                                do_SSSS=True, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
                                do_enhancement=False):
    for S in range(1, all_S + 1):
        for Z in range(1, all_Z + 1):
            print('Now, stitching_CZI: ', path, ' B=', B, ' T=', T, ' S=', S, ' Z=', Z, ' C=', C, ' ')
            stitching_CZI(main_path, path, B, T, S, Z, C, matrix_list, zoom, overlap, output=output, do_SSSS=do_SSSS,
                          name_B=name_B, name_T=name_T, name_S=name_S, name_Z=name_Z, name_C=name_C,
                          do_enhancement=do_enhancement)

    return True


def stitching_CZI_IEed_allC_allZ_bat(main_path, path, B, T, all_S, all_Z, all_C, matrix_list, zoom, overlap,
                                     output=None, do_SSSS=True, name_B=False, name_T=False, name_S=False, name_Z=True,
                                     name_C=True, do_enhancement=False):
    for S in range(1, all_S + 1):
        for Z in range(1, all_Z + 1):
            for C in range(1, all_C + 1):
                print('Now, stitching_CZI: ', path, ' B=', B, ' T=', T, ' S=', S, ' Z=', Z, ' C=', C, ' ')
                stitching_CZI(main_path, path, B, T, S, Z, C, matrix_list, zoom, overlap, output=output,
                              do_SSSS=do_SSSS, name_B=name_B, name_T=name_T, name_S=name_S, name_Z=name_Z,
                              name_C=name_C, do_enhancement=do_enhancement)

    return True


def stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, sp_S, C, matrix_list, zoom, overlap, output=None,
                                         do_SSSS=True, name_C=False):
    # stitching carl zeiss image exported images which had already been existed on the disk
    # !and auto find the best focus Z!
    # input
    # main_path is main path which contain dish_margin.txt
    # path is image path
    # Z, C is which Z, C wanted
    # matrix, zoom, overlap,
    # output=None

    path = os.path.join(main_path, path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    if output is None:
        output = main_path
    else:
        if not os.path.exists(output):
            print('!ERROR! The output does not existed!')
            return False

    # files = os.listdir(path)
    # files.sort
    # for file in files:
    #     pass
    m = 1  # if one tile existed then stitching
    s = sp_S
    t = 1
    demo_z = 1
    if get_CZI_image(path, B, t, s, demo_z, C, m) is not None:  # S while
        while get_CZI_image(path, B, t, s, demo_z, C, m) is not None:  # T while
            print('stitching_CZI_IEed_AutoBestZ_spS_bat(t=', t, ',s=', s, ',C=', C, ',zoom=', zoom, ')')
            stitching_CZI_AutoBestZ(main_path, path, B, t, s, C, matrix_list, zoom, overlap, output=output,
                                    do_SSSS=do_SSSS, name_C=name_C)
            t += 1
        else:
            print('!Warning! : Missing the T:', path, 'S=', s, 'T=', t)
        # s += 1
        # t = 1
    else:
        print('!Warning! : Missing the S:', path, 'S=', s)

    return True


def stitching_CZI_IEed_AutoBestZ_spS_allfolder_bat(main_path, path_father, B, sp_S, C, matrix_list, zoom, overlap,
                                                   output=None, do_SSSS=True, name_C=False):
    # stitching carl zeiss image exported images which had already been existed on the disk
    # !and auto find the best focus Z!
    # input
    # main_path is main path which contain dish_margin.txt
    # path_father is image path's fater folder
    # Z, C is which Z, C wanted
    # matrix, zoom, overlap,
    # output=None

    path_father = os.path.join(main_path, path_father)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(path_father):
        print('!ERROR! The path_father does not existed!')
        return False
    if output is None:
        output = main_path
    else:
        if not os.path.exists(output):
            print('!ERROR! The output does not existed!')
            return False

    for e_f in os.listdir(path_father):

        # !!! CD09 sp code !!!
        if e_f == '2018-09-05~I-5_CD09' or e_f == '2018-09-17~Result_CD09':
            continue
        # !!! CD09 sp code !!!

        each_folder = os.path.join(path_father, e_f)
        if os.path.isdir(each_folder):
            print('stitching_CZI_IEed_AutoBestZ_spS(', each_folder, ', sp_S=', sp_S, ')')

            m = 1  # if one tile existed then stitching
            s = sp_S
            t = 1
            demo_z = 1
            if get_CZI_image(each_folder, B, t, s, demo_z, C, m) is not None:  # S if
                while get_CZI_image(each_folder, B, t, s, demo_z, C, m) is not None:  # T while
                    stitching_CZI_AutoBestZ(main_path, each_folder, B, t, s, C, matrix_list, zoom, overlap,
                                            output=output, do_SSSS=do_SSSS, name_C=name_C)
                    t += 1
                else:
                    print('!Warning! : Missing the T:', each_folder, 'S=', s, 'T=', t)
                # s += 1
                # t = 1
            else:
                print('!Warning! : Missing the S:', each_folder, 'S=', s)

        else:
            print('!NOTICE!', each_folder, 'is a file')

    return True


def merge_all_S_images(path, zoom, to_path='All_images'):
    # merge_all_S_images

    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    to_path = os.path.join(path, to_path)
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    zoom_str = "%.0f%%" % (zoom * 100)
    folder_SSS = 'SSS_' + zoom_str
    folder_SSSS = 'SSSS_' + zoom_str
    S_list = [folder_SSS]  # [folder_SSS, folder_SSSS]

    for i_S in S_list:
        i_S_from = os.path.join(path, i_S)
        if os.path.exists(i_S_from):
            for this_S in os.listdir(i_S_from):
                S_full = os.path.join(i_S_from, this_S)
                S = int(this_S.split('S')[1])
                for from_name in os.listdir(S_full):
                    to_name = 'S%02d~' % S + from_name
                    if not os.path.exists(os.path.join(to_path, i_S)):
                        os.makedirs(os.path.join(to_path, i_S))
                    from_full = os.path.join(S_full, from_name)
                    to_full = os.path.join(to_path, i_S, to_name)
                    shutil.copy(from_full, to_full)

    return True


def extract_spTime_images(path, zoom, spText, to_path=None):
    # spText must be '2018-11-28~IPS_CD13' (from 2018-11-28~IPS_CD13~T7.jpg)

    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    if spText is None or spText == '':
        print('!ERROR! The spText must be str!')
        return False
    if to_path is None or to_path == '':
        to_path = os.path.join(path, spText)
    else:
        to_path = os.path.join(path, to_path)
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    zoom_str = "%.0f%%" % (zoom * 100)
    folder_SSS = 'SSS_' + zoom_str
    folder_SSSS = 'SSSS_' + zoom_str
    S_list = [folder_SSS]  # [folder_SSS, folder_SSSS]

    for i_S in S_list:
        i_S_from = os.path.join(path, i_S)  # main_path/SSS_100%
        if os.path.exists(i_S_from):
            for this_S in os.listdir(i_S_from):  # S1,S2,...
                S_full = os.path.join(i_S_from, this_S)  # main_path/SSS_100%/S1
                S = int(this_S.split('S')[1])
                for from_name in os.listdir(S_full):  # 2018-11-28~IPS_CD13~T7.jpg, 2018-12-01~I-1_CD13~T3.jpg, ...
                    if from_name.find(spText) == 0:
                        to_name = 'S%02d~' % S + from_name
                        if not os.path.exists(os.path.join(to_path, i_S)):
                            os.makedirs(os.path.join(to_path, i_S))
                        from_full = os.path.join(S_full, from_name)
                        to_full = os.path.join(to_path, to_name)
                        shutil.copy(from_full, to_full)

    return True


def move_merge_image_folder(from_path, to_path, zoom, del_from=False):
    # This program is design for merge SSS folder or SSSS folder with specific zoom
    # When one experiment is Separating into multiple block. exp: CD13-1, CD13-2, CD13-3
    # !!! It will replaces the to_path image files using the new files in from_path !!!
    # !!! It will delete the original folder !!!

    if not os.path.exists(from_path):
        print('!ERROR! The from_path does not existed!')
        return False
    if not os.path.exists(to_path):
        print('!ERROR! The to_path does not existed!')
        return False
    if not isinstance(zoom, list):
        zoom = [zoom]

    for i_zoom in zoom:
        zoom_str = "%.0f%%" % (i_zoom * 100)
        folder_SSS = 'SSS_' + zoom_str
        folder_SSSS = 'SSSS_' + zoom_str
        S_list = [folder_SSS, folder_SSSS]

        for i_S in S_list:
            i_S_from = os.path.join(from_path, i_S)
            i_S_to = os.path.join(to_path, i_S)
            if os.path.exists(i_S_from):
                if not os.path.exists(i_S_to):
                    os.makedirs(i_S_to)
                i_each_Sfolder = os.listdir(i_S_from)
                for this_Sfolder in i_each_Sfolder:
                    from_Spath = os.path.join(i_S_from, this_Sfolder)
                    to_Spath = os.path.join(i_S_to, this_Sfolder)
                    if not os.path.exists(to_Spath):
                        os.makedirs(to_Spath)
                    img_files = os.listdir(from_Spath)
                    for img in img_files:
                        img_file = os.path.join(from_Spath, img)  # is a sp image path
                        to_file = os.path.join(to_Spath, img)
                        if os.path.exists(to_file):
                            os.remove(to_file)
                        if del_from:
                            shutil.move(img_file, to_Spath)
                        else:
                            shutil.copy(img_file, to_file)
                if del_from:
                    shutil.rmtree(i_S_from)
    return True


def copy_merge_well_image(from_path, to_path, zoom, S):
    # This program is design for merge SSS folder or SSSS folder with specific zoom
    # When one experiment is Separating into multiple block. exp: CD13-1, CD13-2, CD13-3
    # !!! It will replaces the to_path image files using the new files in from_path !!!

    if not os.path.exists(from_path):
        print('!ERROR! The from_path does not existed!')
        return False
    if not os.path.exists(to_path):
        print('!ERROR! The to_path does not existed!')
        return False
    if not isinstance(zoom, list):
        zoom = [zoom]

    for i_zoom in zoom:
        zoom_str = "%.0f%%" % (i_zoom * 100)
        folder_SSS = 'SSS_' + zoom_str
        folder_SSSS = 'SSSS_' + zoom_str
        S_list = [folder_SSS, folder_SSSS]

        for i_S in S_list:
            i_S_from = os.path.join(from_path, i_S)
            i_S_to = os.path.join(to_path, i_S)
            if os.path.exists(i_S_from):
                if not os.path.exists(i_S_to):
                    os.makedirs(i_S_to)
                i_each_Sfolder = os.listdir(i_S_from)
                for this_Sfolder in i_each_Sfolder:
                    if this_Sfolder == 'S' + str(S):
                        from_Spath = os.path.join(i_S_from, this_Sfolder)
                        to_Spath = os.path.join(i_S_to, this_Sfolder)
                        if not os.path.exists(to_Spath):
                            os.makedirs(to_Spath)
                        img_files = os.listdir(from_Spath)
                        for img in img_files:
                            img_file = os.path.join(from_Spath, img)  # is a sp image path
                            to_file = os.path.join(to_Spath, img)
                            if os.path.exists(to_file):
                                os.remove(to_file)
                            shutil.copy(img_file, to_file)
                # if del_from:
                #     shutil.rmtree(i_S_from)
    return True


def image_pre_treatment_bat(from_path, to_path, zoom, analysis_function, sort_function=None, S=None, do_SSS=True,
                            do_SSSS=True):
    # only generate enhancement img
    # sort_function for file in S folder sorting
    # analysis_function has to input: analysis_function(from_file, to_file)

    if not os.path.exists(from_path):
        print('!ERROR! The from_path does not existed!')
        return False
    to_path = os.path.join(from_path, to_path)
    if not os.path.exists(to_path):
        print('!NOTICE! The to_path does not existed! Now create')
        os.makedirs(to_path)
    else:
        print('!NOTICE! The to_path does existed!')

    if not isinstance(zoom, list):
        zoom = [zoom]
    if (S is None) or S == '':
        pass  # do all
    elif not isinstance(S, list):
        S = [S]

    for i_zoom in zoom:
        zoom_str = "%.0f%%" % (i_zoom * 100)
        folder_SSS = 'SSS_' + zoom_str
        folder_SSSS = 'SSSS_' + zoom_str
        # S_list = [folder_SSS, folder_SSSS]
        S_list = []
        if do_SSS:
            S_list.append(folder_SSS)
        if do_SSSS:
            S_list.append(folder_SSSS)

        for i_S in S_list:
            i_S_from = os.path.join(from_path, i_S)
            i_S_to = os.path.join(to_path, i_S)
            if os.path.exists(i_S_from):
                if not os.path.exists(i_S_to):
                    os.makedirs(i_S_to)
                i_each_Sfolder = os.listdir(i_S_from)
                for this_Sfolder in i_each_Sfolder:
                    S_int = int(this_Sfolder.split('S')[1])
                    if (S is None) or (S == '') or (S_int in S):
                        from_Spath = os.path.join(i_S_from, this_Sfolder)
                        to_Spath = os.path.join(i_S_to, this_Sfolder)
                        if not os.path.exists(to_Spath):
                            os.makedirs(to_Spath)
                        img_files = os.listdir(from_Spath)
                        if sort_function is not None:
                            img_files = sort_function(img_files)
                        for img in img_files:
                            img_file = os.path.join(from_Spath, img)  # is a sp image path
                            to_file = os.path.join(to_Spath, img)
                            if os.path.exists(to_file):
                                os.remove(to_file)

                            # print('>>> my_enhancement(', this_Sfolder, '): ',img_file)
                            # image_my_enhancement(img_file, to_file)
                            print('>>> analysis_function(', this_Sfolder, '): ', img_file)
                            analysis_function(img_file, to_file)
                # if del_from:
                #     shutil.rmtree(i_S_from)
    return True


def image_pre_treatment_bat_old(from_path, to_path, zoom, S=None):
    # This program is design for merge SSS folder or SSSS folder with specific zoom
    # When one experiment is Separating into multiple block. exp: CD13-1, CD13-2, CD13-3
    # !!! It will replaces the to_path image files using the new files in from_path !!!

    if not os.path.exists(from_path):
        print('!ERROR! The from_path does not existed!')
        return False
    if not os.path.exists(to_path):
        print('!ERROR! The to_path does not existed!')
        return False
    if not isinstance(zoom, list):
        zoom = [zoom]
    if (S is None) or S == '':
        pass  # do all
    elif not isinstance(S, list):
        S = [S]

    for i_zoom in zoom:
        zoom_str = "%.0f%%" % (i_zoom * 100)
        folder_SSS = 'SSS_' + zoom_str
        folder_SSSS = 'SSSS_' + zoom_str
        S_list = [folder_SSS, folder_SSSS]

        for i_S in S_list:
            i_S_from = os.path.join(from_path, i_S)
            i_S_to = os.path.join(to_path, i_S)
            if os.path.exists(i_S_from):
                if not os.path.exists(i_S_to):
                    os.makedirs(i_S_to)
                i_each_Sfolder = os.listdir(i_S_from)
                for this_Sfolder in i_each_Sfolder:
                    S_int = int(this_Sfolder.split('S')[1])
                    if (S is None) or (S == '') or (S_int in S):
                        from_Spath = os.path.join(i_S_from, this_Sfolder)
                        to_Spath = os.path.join(i_S_to, this_Sfolder)
                        if not os.path.exists(to_Spath):
                            os.makedirs(to_Spath)
                        img_files = os.listdir(from_Spath)
                        for img in img_files:
                            img_file = os.path.join(from_Spath, img)  # is a sp image path
                            to_file = os.path.join(to_Spath, img)
                            if os.path.exists(to_file):
                                os.remove(to_file)

                            # print('>>> my_enhancement(', this_Sfolder, '): ',img_file)
                            # image_my_enhancement(img_file, to_file)
                            print('>>> my_PGC(', this_Sfolder, '): ', img_file)
                            image_my_PGC(img_file, to_file)
                # if del_from:
                #     shutil.rmtree(i_S_from)
    return True


def rename_CZI_IEXP(path, new_name):
    # rename the path and the image files
    # input the specific path
    # old name: 20180903_1800_I-01_CHIR2-4-6-8-10-12_24H-30H-36H-42H-48H_Tiles109_Z3_Time_5x2_CD09
    # new name: 2018-09-03~I-1_CD09
    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False

    t_xep_namp = os.path.split(path)  # D:\exp\CD09\image_exported\
    # 20180903_1800_I-01_CHIR2-4-6-8-10-12_24H-30H-36H-42H-48H_Tiles109_Z3_Time_5x2_CD09
    old_name = t_xep_namp[1]

    for this_file in os.listdir(path):
        this_new_name = new_name + '~' + this_file.split(old_name)[1][1:]
        os.rename(os.path.join(path, this_file), os.path.join(path, this_new_name))

    os.rename(path, os.path.join(t_xep_namp[0], new_name))

    return True


def rename_S_number(path, shift):
    # rename the S path name
    # input the specific S path
    # shift: exp: +24
    # example: S71 add 24 to S95

    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False

    files = os.listdir(path)

    if shift > 0:
        files.sort(key=lambda x: [int(x.split('S=')[1])], reverse=True)
    else:
        files.sort(key=lambda x: [int(x.split('S=')[1])], reverse=False)

    for old_S_name in files:
        new_S_name = 'S=' + str(int(old_S_name.split('S=')[1]) + shift)
        os.rename(os.path.join(path, old_S_name), os.path.join(path, new_S_name))

    return True


def make_empty_SZC(path, S, Z, C):
    # rename the S path name
    # input the specific S path
    # shift: exp: +24
    # example: S71 add 24 to S95

    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False

    for s in range(0, S):
        s_path = os.path.join(path, 'S=' + str(s))
        os.makedirs(s_path)
        for z in range(0, Z):
            z_path = os.path.join(s_path, 'Z=' + str(z))
            os.makedirs(z_path)
            for c in range(0, C):
                c_path = os.path.join(z_path, 'C=' + str(c))
                os.makedirs(c_path)

    return True


def add_prefix(image_path, prefix):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    for i in path_list:  # r'Label_1.png'
        old_img_file = os.path.join(image_path, i)
        new_name = prefix + '~' + i
        new_img_file = os.path.join(image_path, new_name)
        os.rename(old_img_file, new_img_file)

    return True


def remove_suffix(image_path, suffix):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    # for i in path_list:  # r'Label_1.png'
    #     old_img_file = os.path.join(image_path, i)
    #     new_name = prefix + '~' + i
    #     new_img_file = os.path.join(image_path, new_name)
    #     os.rename(old_img_file, new_img_file)

    return True


def remove_suffix_2(image_path):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    for i in path_list:  # r'Label_1.png'
        old_img_file = os.path.join(image_path, i)
        new_name = i.split('~')[0] + '~' + i.split('~')[1] + '.' + i.split('.')[-1]
        new_img_file = os.path.join(image_path, new_name)
        os.rename(old_img_file, new_img_file)

    return True


def put_img_into_auto_export_position_forCD26(main_path, date, exp_name):
    # for CD26 EXP error!

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    date_path = os.path.join(main_path, date)
    if not os.path.exists(date_path):
        os.makedirs(date_path)

    exp_name_path = os.path.join(date_path, exp_name)
    if not os.path.exists(exp_name_path):
        os.makedirs(exp_name_path)

    B_path = os.path.join(exp_name_path, 'B')
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    image_file_type = ('.jpg', '.png', '.tif')

    files = os.listdir(main_path)
    for file in files:
        if file[-4:] in image_file_type:
            if file.split('_S00')[0] == exp_name:
                img_path = os.path.join(main_path, file)
                this_S = int(file.split('_S00')[1].split('(')[0])
                this_T = int(file.split('_T')[1].split('_Z')[0])
                this_Z = int(file.split('_Z')[1].split('_C')[0])
                this_C = int(file.split('_C')[1].split('_M')[0])
                this_M = int(file.split('_M')[1].split('_ORG')[0])
                to_path = os.path.join(B_path, 'T=' + str(this_T), 'S=' + str(this_S), 'Z=' + str(this_Z),
                                       'C=' + str(this_C))
                if not os.path.exists(to_path):
                    os.makedirs(to_path)
                shutil.move(img_path, to_path)

    return True


def put_img_into_auto_export_position_forCD27(main_path, date, exp_name):
    # for CD26 EXP error!

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    date_path = os.path.join(main_path, date)
    if not os.path.exists(date_path):
        os.makedirs(date_path)

    exp_name_path = os.path.join(date_path, exp_name)
    if not os.path.exists(exp_name_path):
        os.makedirs(exp_name_path)

    B_path = os.path.join(exp_name_path, 'B')
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    image_file_type = ('.jpg', '.png', '.tif')

    files = os.listdir(main_path)
    for file in files:
        if file[-4:] in image_file_type:
            if file.split('_S00')[0] == exp_name:
                img_path = os.path.join(main_path, file)
                this_S = int(file.split('_S00')[1].split('(')[0])
                # this_T = int(file.split('_T')[1].split('_Z')[0])
                # this_Z = int(file.split('_Z')[1].split('_C')[0])
                this_C = int(file.split('_C')[1].split('_M')[0])
                this_M = int(file.split('_M')[1].split('_ORG')[0])
                to_path = os.path.join(B_path, 'S=' + str(this_S), 'C=' + str(this_C))
                if not os.path.exists(to_path):
                    os.makedirs(to_path)
                shutil.move(img_path, to_path)

    return True


def put_img_into_auto_export_position_forCD41_1(main_path, date, exp_name):
    # for CD41 EXP error!
    # main_path = r'H:\CD41\PROCESSING'
    # date = r'2020-01-16' # date = r'2020-01-16'
    # exp_name = r'CD41_IPS' # exp_name = r'CD41_STAGEI_1H'
    # EXP: 'CD41_IPS_S0000(A1-A1)_Z0000_C00_M0014_ORG'
    # EXP: 'CD41_STAGEI_39H_S0000(A1-A1)_T000000_Z0000_C00_M0005_ORG.tif'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    date_path = os.path.join(main_path, date)
    if not os.path.exists(date_path):
        os.makedirs(date_path)

    exp_name_path = os.path.join(date_path, exp_name)
    if not os.path.exists(exp_name_path):
        os.makedirs(exp_name_path)

    B_path = os.path.join(exp_name_path, 'B')
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    image_file_type = ('.jpg', '.png', '.tif')

    files = os.listdir(main_path)
    for file in files:
        if file[-4:] in image_file_type:
            if file.split('_S00')[0] == exp_name:
                img_path = os.path.join(main_path, file)
                this_S = int(file.split('_S00')[1].split('(')[0])
                # this_T = int(file.split('_T')[1].split('_Z')[0])
                this_Z = int(file.split('_Z')[1].split('_C')[0])
                this_C = int(file.split('_C')[1].split('_M')[0])
                this_M = int(file.split('_M')[1].split('_ORG')[0])
                to_path = os.path.join(B_path, 'S=' + str(this_S), 'Z=' + str(this_Z), 'C=' + str(this_C))
                if not os.path.exists(to_path):
                    os.makedirs(to_path)
                shutil.move(img_path, to_path)

    return True


def put_img_into_auto_export_position_forCD41_2(main_path, date, exp_name):
    # for CD41 EXP error!
    # main_path = r'H:\CD41\PROCESSING'
    # date = r'2020-01-16' # date = r'2020-01-16'
    # exp_name = r'CD41_IPS' # exp_name = r'CD41_STAGEI_1H'
    # EXP: 'CD41_IPS_S0000(A1-A1)_Z0000_C00_M0014_ORG'
    # EXP: 'CD41_STAGEI_39H_S0000(A1-A1)_T000000_Z0000_C00_M0005_ORG.tif'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    date_path = os.path.join(main_path, date)
    if not os.path.exists(date_path):
        os.makedirs(date_path)

    exp_name_path = os.path.join(date_path, exp_name)
    if not os.path.exists(exp_name_path):
        os.makedirs(exp_name_path)

    B_path = os.path.join(exp_name_path, 'B')
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    image_file_type = ('.jpg', '.png', '.tif')

    files = os.listdir(main_path)
    for file in files:
        if file[-4:] in image_file_type:
            if file.split('_S00')[0] == exp_name:
                img_path = os.path.join(main_path, file)
                this_S = int(file.split('_S00')[1].split('(')[0])
                this_T = int(file.split('_T')[1].split('_Z')[0])
                this_Z = int(file.split('_Z')[1].split('_C')[0])
                this_C = int(file.split('_C')[1].split('_M')[0])
                this_M = int(file.split('_M')[1].split('_ORG')[0])
                to_path = os.path.join(B_path, 'T=' + str(this_T), 'S=' + str(this_S), 'Z=' + str(this_Z),
                                       'C=' + str(this_C))
                if not os.path.exists(to_path):
                    os.makedirs(to_path)
                shutil.move(img_path, to_path)

    return True


def extract_CD13_key_frames(main_path, to_path):
    # main_path = r'C:\C137\PROCESSING\CD13\SSSS_100%'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    if not os.path.exists(to_path):
        os.makedirs(to_path)

    file_names = ["2018-11-30~IPS-3_CD13~T18.jpg", "2018-12-01~I-1_CD13~T1.jpg", "2018-12-01~I-1_CD13~T2.jpg",
                  "2018-12-01~I-1_CD13~T3.jpg", "2018-12-01~I-1_CD13~T4.jpg", "2018-12-01~I-1_CD13~T5.jpg",
                  "2018-12-01~I-1_CD13~T6.jpg", "2018-12-01~I-1_CD13~T7.jpg", "2018-12-01~I-1_CD13~T8.jpg",
                  "2018-12-01~I-1_CD13~T9.jpg", "2018-12-01~I-1_CD13~T10.jpg"]

    S_folders = os.listdir(main_path)
    for folder in S_folders:
        S_path = os.path.join(main_path, folder)
        img_files = os.listdir(S_path)
        for file in img_files:
            if file in file_names:
                from_img = os.path.join(S_path, file)
                to_S = os.path.join(to_path, folder)
                if not os.path.exists(to_S):
                    os.makedirs(to_S)
                to_full = os.path.join(to_S, file)
                shutil.copy(from_img, to_full)

    return True


# def image_enhancement_bat(main_path, zoom, sort_function, analysis_function, S=None, do_SSS=True, do_SSSS=True):
#     # only generate enhancement img
#     # sort_function for file in S folder sorting
#     # analysis_function has to input: analysis_function(from_file, to_file)
#
#     if not os.path.exists(main_path):
#         print('!ERROR! The main_path does not existed!')
#         return False
#
#     folder_MyPGC_img = 'MyPGC_img'
#     zoom_str = "%.0f%%" % (zoom * 100)
#     if (S is None) or S == '':
#         pass  # do all
#     elif not isinstance(S, list):
#         S = [S]
#
#     SSS_path = os.path.join(main_path, 'SSS_' + zoom_str)
#     SSSS_path = os.path.join(main_path, 'SSSS_' + zoom_str)
#     has_SSS = True
#     has_SSSS = True
#
#     if not os.path.exists(SSS_path):
#         print('!Caution! The', SSS_path, 'does not existed!')
#         has_SSS = False
#     if not os.path.exists(SSSS_path):
#         print('!Caution! The', SSSS_path, 'does not existed!')
#         has_SSSS = False
#
#     if do_SSS and has_SSS:
#         if S is None:
#             path_list = os.listdir(SSS_path)
#             path_list.sort(key=lambda x: int(x.split('S')[1]))
#             for this_S_folder in path_list:  # S1 to S96
#                 Spath = os.path.join(SSS_path, this_S_folder)
#                 img_files_list = os.listdir(Spath)
#                 sort_function(img_files_list)
#                 for img in img_files_list:
#                     analysis_function(from_file, to_file)
#         else:
#             S
#
#     if do_SSSS and has_SSSS:
#         path_list = os.listdir(SSSS_path)
#         path_list.sort(key=lambda x: int(x.split('S')[1]))
#         for this_S_folder in path_list:  # S1 to S96
#             Spath = os.path.join(SSSS_path, this_S_folder)
#             img_files_list = os.listdir(Spath)
#             sort_function(img_files_list)
#             for img in img_files_list:
#                 analysis_function(from_file, to_file)
#
#
#     return True


# def temp_1():
#     input = r'E:\CD13_Enhance\SSS_100%'
#     S = 96
#     for s in range(34, 34 + 1):
#         dst = os.path.join(input, 'S' + str(s))#
#         # from_file = get_CZI_image(r'E:\CD13_result', 1, 1, s, 1, 1, 1)[0]
#         # to_file = os.path.join(dst, os.path.split(from_file)[1])
#         # image_my_enhancement(from_file, to_file)
#         #
#         # from_file = get_CZI_image(r'E:\CD13_result', 1, 1, s, 1, 2, 1)[0]
#         # to_file = os.path.join(dst, os.path.split(from_file)[1])
#         # image_my_enhancement(from_file, to_file)
#
#         from_file = get_CZI_image(r'E:\CD13_result', 1, 1, s, 1, 3, 1)[0]
#         to_file = os.path.join(dst, os.path.split(from_file)[1])
#         image_my_enhancement_sub_mode(from_file, to_file)
#         # image_my_enhancement_sub_mode(to_file, to_file)


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Lib_Function.py !')

    main_path = r'C:\Users\Kitty\Desktop\CD46'
    path = r'C:\Users\Kitty\Desktop\CD46\CD46_Stage-1_1H'

    stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_25_Tiles(), [1], 0.05, T=1, S=1, output=None,
                                     do_SSSS=True, name_C=False)

    # folder_image_resize(r'C:\Users\Kitty\Desktop\cTnT_2580\A\train', size=(2580, 2580))
    # folder_image_resize(r'C:\Users\Kitty\Desktop\cTnT_2580\B\train', size=(2580, 2580))
    # image_path=r'C:\Users\Kitty\Desktop\cTnT_2580\A\train'
    # output_path=r'C:\Users\Kitty\Desktop\cTnT_860\A\train'
    # folder_image_cut_n_blocks(image_path, output_path, n=3)
    # image_path=r'C:\Users\Kitty\Desktop\cTnT_2580\B\train'
    # output_path=r'C:\Users\Kitty\Desktop\cTnT_860\B\train'
    # folder_image_cut_n_blocks(image_path, output_path, n=3)

    # remove_suffix_2(r'C:\Users\Kitty\Desktop\1')
    # remove_suffix_2(r'C:\Users\Kitty\Desktop\2')
    #
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD11\hand_labeling_Day5', 'CD11')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD11\hand_labeling_End', 'CD11')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD11\hand_labeling_End_enhanced', 'CD11')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD11\hand_labeling_IF', 'CD11')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD11\hand_labeling_IF_enhanced', 'CD11')
    #
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD13\hand_labeling_Day5', 'CD13')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD13\hand_labeling_End', 'CD13')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD13\hand_labeling_End_enhanced', 'CD13')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD13\hand_labeling_IF', 'CD13')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD13\hand_labeling_IF_enhanced', 'CD13')

    # folder_image_resize(r'C:\Users\Kitty\Desktop\cTnT_2580\A\train', size=(2580, 2580))
    # folder_image_resize(r'C:\Users\Kitty\Desktop\cTnT_2580\B\train', size=(2580, 2580))

    # image_path=r'C:\Users\Kitty\Desktop\cTnT_2580\A\train'
    # output_path=r'C:\Users\Kitty\Desktop\cTnT_860\A\train'
    # folder_image_cut_n_blocks(image_path, output_path, n=3)
    # image_path=r'C:\Users\Kitty\Desktop\cTnT_2580\B\train'
    # output_path=r'C:\Users\Kitty\Desktop\cTnT_860\B\train'
    # folder_image_cut_n_blocks(image_path, output_path, n=3)

    # remove_suffix_2(r'C:\Users\Kitty\Desktop\cTnT\A\train')
    # remove_suffix_2(r'C:\Users\Kitty\Desktop\cTnT\B\train')
    # folder_image_resize(r'C:\Users\Kitty\Desktop\cTnT_2480\A\train', size=(2480, 2480))
    # folder_image_resize(r'C:\Users\Kitty\Desktop\cTnT_2480\B\train', size=(2480, 2480))

    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58A\hand_labeling_Day5', 'CD58A')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58A\hand_labeling_End', 'CD58A')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58A\hand_labeling_End_enhanced', 'CD58A')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58A\hand_labeling_IF', 'CD58A')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58A\hand_labeling_IF_enhanced', 'CD58A')
    #
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58B\hand_labeling_Day5', 'CD58B')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58B\hand_labeling_End', 'CD58B')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58B\hand_labeling_End_enhanced', 'CD58B')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58B\hand_labeling_IF', 'CD58B')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD58B\hand_labeling_IF_enhanced', 'CD58B')
    #
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD61\hand_labeling_Day5', 'CD61')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD61\hand_labeling_End', 'CD61')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD61\hand_labeling_End_enhanced', 'CD61')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD61\hand_labeling_IF', 'CD61')
    # add_prefix(r'E:\Image_Processing\hand_labeling\CD61\hand_labeling_IF_enhanced', 'CD61')

    # main_path = r'D:\CD58\Processing'
    # path = r'D:\CD58\Processing\2021-04-16\CD58A2_IF'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, 1, 3, return_96well_25_Tiles(), 1, 0.16, output=None,
    #                                       do_SSSS=True)

    # main_path = r'I:\CD44\PROCESSING'
    # B = 1
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.1
    # path = r'D:\CD11\2018-11-06~IPS_0_CD11'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=36, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)

    # main_path = r'I:\CD44\PROCESSING'
    # path = r'I:\CD44\PROCESSING\2020-07-23\CD44_Result-IF'
    # B = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.05
    # all_C = 3
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)

    # main_path = r'J:\CD43\PROCESSING'
    # path = r'J:\CD43\PROCESSING\MyPGC_img\SSS_100%'
    # row = 8
    # col = 12
    # output = 'All_Wells'
    # stage_str = 'IPS-1'
    # T = 1
    # stitching_well_one_picture(main_path, path, output, row, col, stage_str, T, w=320, h=320, zoom=None,
    #                            sort_function=None)
    # stitching_well_all_time(main_path, path, row, col, w=960, h=960, zoom=None, output='All_Wells',
    #                         sort_function=files_sort_CD42)

    # print('>>>NOW rename:')

    # path = r'D:\CD11\I_1_CD11'
    # rename_CZI_IEXP(path, r'2018-11-08~I_1_CD11')
    # print('>>>FINISHED! rename')

    # B = 1
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.1
    # path = r'D:\CD11\2018-11-06~IPS_0_CD11'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=36, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)
    #
    # path = r'D:\CD11\2018-11-08~I_1_CD11'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)

    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, 1, 0.1, output=None, do_SSSS=True,
    #                                  name_C=False)
    #
    # path = r'C:\C137\PROCESSING\CD11\2018-11-08\STAGE_1_CD11'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, 1, 0.1, output=None, do_SSSS=True,
    #                                  name_C=False)

    # main_path = r'C:\C137\PROCESSING\CD13\SSSS_100%'
    # to_path = r'C:\Users\Kitty\Desktop\CD13\SSSS_100%'
    # extract_CD13_key_frames(main_path, to_path)

    # main_path = r'G:\CD40\processing'
    # path = r'G:\CD40\processing\2020-01-20\CD40(A)_RESULT(beating)'
    # B = 1
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.05
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    #                                  name_C=True)
    #
    # path = r'G:\CD40\processing\2020-01-21\CD40(B)_RESULT(beating)'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    #                                  name_C=True)
    #
    # all_C = 3
    # path = r'G:\CD40\processing\2020-02-08\CD40(A)_RESULT(CTNT)'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)
    #
    # path = r'G:\CD40\processing\2020-02-09\CD40(B)_RESULT(CTNT)'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)

    # date = r'2020-01-16'  # date = r'2020-01-16'
    # exp_name = r'CD41_IPS'  # exp_name = r'CD41_STAGEI_1H'
    # put_img_into_auto_export_position_forCD41_1(main_path, date, exp_name)
    # date = r'2020-01-16'
    # exp_name = r'CD41_STAGEI_1H'
    # put_img_into_auto_export_position_forCD41_2(main_path, date, exp_name)

    # put_img_into_auto_export_position_forCD27(r'G:\CD27\Processing', '2019-07-25', 'CD27_Result_IF')

    # main_path = r'D:\MLC\Goose'
    # path = r'D:\MLC\Goose\2019-11-14\Exp1_Day2'
    # B = 1
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.05
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
    #                                  name_C=False)

    # path = r'D:\PROCESSING\CD13'
    # zoom = 1
    # spText = '2018-12-10~III-1_CD13~T44'
    # extract_spTime_images(path, zoom, spText, to_path='End')
    # spText = '2018-12-10~III-1_CD13~T45'
    # extract_spTime_images(path, zoom, spText, to_path='End')

    # rename_CZI_IEXP(r'E:\CD13\PROCESSING\result_parallel_CD13', 'Result_CD13')
    # from_path = r'E:\CD13\PROCESSING'
    # # image_pre_treatment_bat(from_path, 'MyPGC_img', 1, image_my_PGC, sort_function=files_sort_CD13, S=44)
    # image_pre_treatment_bat(from_path, 'MyPGC_img', 1, image_my_PGC, sort_function=files_sort_CD13,
    #                         S=[14, 15, 16, 17, 18, 19, 20])
    # image_pre_treatment_bat(from_path, 'MyPGC_img', 1, image_my_PGC, sort_function=files_sort_CD13,
    #                         S=[21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    # image_pre_treatment_bat(from_path, 'MyPGC_img', 1, image_my_PGC, sort_function=files_sort_CD13,
    #                         S=[31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
    # image_pre_treatment_bat(from_path, 'MyPGC_img', 1, image_my_PGC, sort_function=files_sort_CD13,
    #                         S=[41, 42, 43, 45, 46, 47, 48, 49, 50])
    # image_pre_treatment_bat(from_path, 'MyPGC_img', 1, image_my_PGC, sort_function=files_sort_CD13,
    #                         S=[51, 52, 53, 54, 55, 56, 57, 58, 59, 60])
    # image_pre_treatment_bat(from_path, 'MyPGC_img', 1, image_my_PGC, sort_function=files_sort_CD13,
    #                         S=[61, 62, 63, 64, 65, 66, 67, 68, 69, 70])
    # image_pre_treatment_bat(from_path, 'MyPGC_img', 1, image_my_PGC, sort_function=files_sort_CD13,
    #                         S=[71, 72, 73, 74, 75, 76, 77, 78, 79, 80])

    # stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 15, C, matrix_list, zoom, overlap, output=None,
    #                                      do_SSSS=False, name_C=False)
    # stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 16, C, matrix_list, zoom, overlap, output=None,
    #                                      do_SSSS=False, name_C=False)
    # stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 18, C, matrix_list, zoom, overlap, output=None,
    #                                      do_SSSS=False, name_C=False)
    # stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 19, C, matrix_list, zoom, overlap, output=None,
    #                                      do_SSSS=False, name_C=False)

    # output = r'C:\C137\PROCESSING\CD09\MyPGC_img\All_wells_10%'
    # stitching_well(main_path, path, 4, 6, w=0, h=0, zoom=0.2, output=output, sort_function=files_sort_CD09)
    # output = r'C:\C137\PROCESSING\CD09\MyPGC_img\All_wells_25%'
    # stitching_well(main_path, path, 4, 6, w=0, h=0, zoom=0.5, output=output, sort_function=files_sort_CD09)

    # main_path = r'E:\CD26\Processing'
    # path = r'E:\CD26\Processing\MyPGC_img\SSS_100%'
    # output = r'E:\CD26\Processing\MyPGC_img\All_wells_1%'
    # stitching_well(main_path, path, 8, 12, w=0, h=0, zoom=0.01, output=output)
    # output = r'E:\CD26\Processing\MyPGC_img\All_wells_5%'
    # stitching_well(main_path, path, 8, 12, w=0, h=0, zoom=0.05, output=output)

    # main_path = r'L:\CD31\PROCESSING'
    # path = r'L:\CD31\PROCESSING\MyPGC_img\SSS_100%'
    # output = r'L:\CD31\PROCESSING\MyPGC_img\All_wells_960'
    # stitching_well(main_path, path, 8, 12, w=960, h=960, output=output)
    # output = r'L:\CD31\PROCESSING\MyPGC_img\All_wells_2430'
    # stitching_well(main_path, path, 8, 12, w=2430, h=2430, output=output)
    #
    # main_path = r'L:\CD32\PROCESSING'
    # path = r'L:\CD32\PROCESSING\MyPGC_img\SSS_100%'
    # output = r'L:\CD32\PROCESSING\MyPGC_img\All_wells_960'
    # stitching_well(main_path, path, 8, 12, w=960, h=960, output=output)
    # output = r'L:\CD32\PROCESSING\MyPGC_img\All_wells_2430'
    # stitching_well(main_path, path, 8, 12, w=2430, h=2430, output=output)

    # path = r'E:\CD30\PROCESSING\Enhanced_img'
    # merge_all_S_images(path, 1, to_path='All_images')
    # path = r'E:\CD30\PROCESSING\MyPGC_img'
    # merge_all_S_images(path, 1, to_path='All_images')

    # main_path = r'L:\CD33\PROCESSING'
    # path = r'L:\CD33\PROCESSING\2019-08-05\CD33_STAGE3_270H_[Beating]'
    # B = 1
    # C = 1
    # matrix_list = return_96well_30_Tiles()
    # zoom = [1]
    # overlap = 0.05
    #
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    #                                  name_C=False)
    #
    # path = r'L:\CD33\PROCESSING\2019-08-10\CD33_STAGE3_280H_[Result_IF]'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, 1, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    #                                  name_C=True)
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, 2, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    #                                  name_C=True)
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, 3, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    #                                  name_C=True)
    # path = r'L:\CD30\PROCESSING\2019-07-16\CD30(B)_STAGE1_24H_[]\B\T=0'
    # make_empty_SZC(path, 24, 3, 1)
    # path = r'L:\CD30\PROCESSING\2019-07-16\CD30(B)_STAGE1_24H_[]\B\T=1'
    # make_empty_SZC(path, 24, 3, 1)
    # path = r'L:\CD30\impro\2019-07-17\CD30(A)_STAGE1_60H_[]\B\T=0'
    # make_empty_SZC(path, 48, 3, 1)
    # path = r'L:\CD30\impro\2019-07-17\CD30(A)_STAGE1_60H_[]\B\T=1'
    # make_empty_SZC(path, 48, 3, 1)
    # path = r'L:\CD30\impro\2019-07-17\CD30(B)_STAGE1_48H_[]\B\T=0'
    # make_empty_SZC(path, 48, 3, 1)
    # path = r'L:\CD30\impro\2019-07-17\CD30(B)_STAGE1_48H_[]\B\T=1'
    # make_empty_SZC(path, 48, 3, 1)

    # path = r'L:\CD30\PROCESSING\2019-07-16\CD30(B)_STAGE1_24H_[]\B\T=0'
    # rename_S_number(path, 24)
    # path = r'L:\CD30\PROCESSING\2019-07-16\CD30(B)_STAGE1_24H_[]\B\T=1'
    # rename_S_number(path, 24)
    # path = r'L:\CD30\PROCESSING\2019-07-17\CD30(A)_STAGE1_60H_[]\B\T=0'
    # rename_S_number(path, 48)
    # path = r'L:\CD30\PROCESSING\2019-07-17\CD30(A)_STAGE1_60H_[]\B\T=1'
    # rename_S_number(path, 48)
    # path = r'L:\CD30\PROCESSING\2019-07-17\CD30(B)_STAGE1_48H_[2-half]\B\T=0'
    # rename_S_number(path, 48)
    # main_path = r'D:\PROCESSING\CD26'
    # output = r'F:\CD26\Processing'
    # path = r'D:\PROCESSING\CD26\2019-06-29\CD26_Result_IF'
    # # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    # #                        name_C=False)
    # stitching_CZI_IEed_bat(main_path, path, 1, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output, do_SSSS=True,
    #                        name_C=True)
    # stitching_CZI_IEed_bat(main_path, path, 1, 1, 2, return_96well_30_Tiles(), [1], 0.05, output=output, do_SSSS=True,
    #                        name_C=True)
    # stitching_CZI_IEed_bat(main_path, path, 1, 1, 3, return_96well_30_Tiles(), [1], 0.05, output=output, do_SSSS=True,
    #                        name_C=True)
    # path = r'D:\PROCESSING\CD26\2019-06-14\CD26_IPS(H9)'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-14\CD26_STAGEI_0H'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-15\CD26_STAGEI_18H'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-15\CD26_STAGEI_24H'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-15\CD26_STAGEI_30H'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-16\CD26_STAGEI_36H'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-16\CD26_STAGEI_42H'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-16\CD26_STAGEI_48H'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-17\CD26_STAGEII_IWR1'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-19\CD26_STAGEII_D5'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\PROCESSING\CD26\2019-06-27\CD26_End'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, 1, 1, return_96well_30_Tiles(), [1], 0.05, output=output,
    #                                  do_SSSS=True, name_C=False)

    # main_path = r'D:\PROCESSING\CD26'
    # date = r'2019-06-16'
    # exp_name = r'CD26_STAGEI_48H'
    # put_img_into_auto_export_position(main_path, date, exp_name)
    # date = r'2019-06-17'
    # exp_name = r'CD26_STAGEII_IWR1'
    # put_img_into_auto_export_position(main_path, date, exp_name)
    # date = r'2019-06-19'
    # exp_name = r'CD26_STAGEII_D5'
    # put_img_into_auto_export_position(main_path, date, exp_name)

    # main_path = r'D:\PROCESSING\CD23'
    # B = 1
    # Z = 1
    # C = 1
    # matrix_list = return_CD23_Tiles()
    # zoom = 1
    # overlap = 0.05
    # path = r'D:\PROCESSING\CD23\2019-05-22\CD23_A-72h'
    # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True)
    # path = r'D:\PROCESSING\CD23\2019-05-22\CD23_B-72h'
    # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True)
    # path = r'D:\PROCESSING\CD23\2019-05-22\CD23_C-72h'
    # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True)
    #
    # path = r'D:\PROCESSING\CD23\2019-05-24\CD23_A_D4end'
    # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True)
    # path = r'D:\PROCESSING\CD23\2019-05-24\CD23_B_D4end'
    # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True)
    # path = r'D:\PROCESSING\CD23\2019-05-24\CD23_C_D4end'
    # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True)
    #
    # path = r'D:\PROCESSING\CD23\2019-05-25\CD23_A_D5end'
    # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True)
    # path = r'D:\PROCESSING\CD23\2019-05-25\CD23_B_D5end'
    # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True)
    # path = r'D:\PROCESSING\CD23\2019-05-25\CD23_C_D5end'
    # stitching_CZI_IEed_bat(main_path, path, B, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True)
    # temp_1()
    # print(get_CZI_image(r'E:\CD13_result', 1, 1, 21, 1, 3, 1))
    # rename_CZI_IEXP(r'E:\result_parallel_CD13', '2018-12-11~Result_CD13')
    # main_path = r'F:\PROCESSING\CD13'
    # path_father = r'F:\CD09\Exported'
    # path = r'F:\CD09\Exported'
    # sp_S = 10
    # matrix_list = return_CD09_ORCA_Tiles()
    # zoom = [1, 0.5, 0.25, 0.125]
    # from_path = main_path
    # to_path = r'D:\CD09'
    # to_path = r'C:\Users\sunqi\Desktop\CD13_Black'
    # image_pre_treatment_bat(from_path, to_path, 1, S=[3, 4, 5, 6, 7])

    # stitching_CZI(r'C:\Users\Kitty\Desktop\CD22', r'C:\Users\Kitty\Desktop\CD22\2019-05-18\IPS-1_CD22', 1, 1, 2, 2, 1,
    #               return_CD22_Tiles(), 1, 0.05, output=None, do_SSSS=True)

    # new_stitching_CZI_AutoBestZ_(r'C:\Users\Kitty\Desktop\CD22', r'C:\Users\Kitty\Desktop\CD22\2019-05-18\IPS-1_CD22',
    #                              1, 1, 2, 1, return_CD22_Tiles(), 1, 0.05, output=None, do_SSSS=True)
    # image_my_PGC(r'C:\Users\Kitty\Desktop\CD22\SSS_100%\S1\2019-05-18~IPS-1_CD22~T1.png', r'C:\Users\Kitty\Desktop\CD22\SSS_100%\S1\2019-05-18~IPS-1_CD22~T1~PGC.png', show_hist=False)
    # image_my_enhancement(r'C:\Users\Kitty\Desktop\CD22\SSS_100%\S1\2019-05-18~IPS-1_CD22~T1.png', r'C:\Users\Kitty\Desktop\CD22\SSS_100%\S1\2019-05-18~IPS-1_CD22~T1~Enhancement2.png', show_hist=False)
    # image_pre_treatment_bat(from_path, to_path, 1)
    # pic_path = r'C:\Users\sunqi\Desktop\CD13\SSS_100%\S1'
    # output_file = r'C:\Users\sunqi\Desktop\CD13\Video\SSS_S1.1024.avi'
    # make_movies(pic_path, output_file, 1024, 1024, fps=3, color=True)
    # output_file = r'C:\Users\sunqi\Desktop\CD13\Video\SSS_S13.2160.avi'
    # make_movies(pic_path, output_file, 2160, 2160, fps=3, color=True)
    # pic_path =r'C:\Users\Kitty\Desktop\TEMP\S1'
    # output_file =r'C:\Users\Kitty\Desktop\TEMP\S1_test.avi'
    # make_movies(pic_path, output_file, 1024, 1024, fps=3, color=True)

    # image_pre_treatment_bat(from_path, to_path, [0.5, 1])
    # image_my_PGC(r'C:\Users\Kitty\Desktop\TEMP\2018-11-28~IPS_CD13~T1.jpg',
    #              r'C:\Users\Kitty\Desktop\TEMP\19.jpg')
    # image_my_enhancement(r'C:\Users\Kitty\Desktop\TEMP\2018-11-28~IPS_CD13~T1.jpg',  r'C:\Users\Kitty\Desktop\TEMP\25.jpg')

    # image_my_PGC(r'C:\Users\Kitty\Desktop\TEMP\20.jpg', r'C:\Users\Kitty\Desktop\TEMP\23.jpg')
    # copy_merge_well_image(to_path, output, zoom, 1)
    # move_merge_image_folder(output, to_path, zoom, del_from=True)

    # stitching_CZI_IEed_AutoBestZ_spS_allfolder(main_path, path_father, B, sp_S, C, matrix_list, zoom, overlap, output=None)
    # stitching_CZI_IEed_AutoBestZ_spS_allfolder(main_path, path_father, 1, sp_S, 1, matrix_list, zoom, 0.1,
    #                                            output=output, do_SSSS=False)
    # stitching_CZI_IEed_AutoBestZ_spS(main_path, path, B, sp_S, C, matrix_list, zoom, overlap, output=None)
    # stitching_CZI_IEed_AutoBestZ_spS(main_path, path, 1, sp_S, 1, matrix_list, zoom, 0.1, output=output)

    # stitching_CZI_IEed_AutoBestZ_spS_allfolder(main_path, path_father, B, sp_S, C, matrix_list, zoom, overlap, output=None)
    # stitching_CZI_IEed_AutoBestZ_spS_allfolder(main_path, path_father, 1, sp_S, 1, matrix_list, zoom, 0.1, output=output)

    # main_path = r'T:'
    # path = r'T:\2018-09-04~I-2_CD09'
    # matrix_list = return_CD09_ORCA_Tiles()
    # zoom = [1, 0.5, 0.25, 0.125]
    # output = r'D:\TimeSerise\CD09'
    # # stitching_CZI_IEed_AutoBestZ(main_path, path, B, C, matrix_list, zoom, overlap, output=None)
    # stitching_CZI_IEed_AutoBestZ(main_path, path, 1, 1, matrix_list, zoom, 0.1, output=output)
    # path = r'2018-09-13~F_CD09'
    # stitching_CZI_IEed_AutoBestZ(main_path, path, 1, 1, matrix_list, zoom, 0.1, output=output)

    # path = r'F:\CD09\Exported\2018-09-05~I-5_CD09'
    # matrix_list = return_CD09_506_Tiles()
    # zoom = [1, 0.5, 0.25, 0.125]
    # # stitching_CZI_IEed_AutoBestZ(main_path, path, B, C, matrix_list, zoom, overlap, output=None)
    # stitching_CZI_IEed_AutoBestZ(main_path, path, 1, 1, matrix_list, zoom, 0.1, output=output)
    #
    # path = r'F:\CD09\Exported\2018-09-17~Result_CD09'
    # # stitching_CZI_IEed_AutoBestZ(main_path, path, B, C, matrix_list, zoom, overlap, output=None)
    # stitching_CZI_IEed_AutoBestZ(main_path, path, 1, 4, matrix_list, zoom, 0.1, output=output)

    # stitching_CZI_AutoBestZ(main_path, r'F:\CD09\Exported\2018-09-03~I-1_CD09', 1, 10, 1, 1, matrix_list, zoom, 0.1,
    #                         output=output)
    # img_path = r'C:\Users\Kitty\Desktop\NGC7293_test_temp\2018-11-28~IPS_CD13~T1.jpg'
    # img = cv2.imread(img_path, 0)
    # cv2.imshow('input_image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(return_CD13_Tiles())
    # path = r'X:\20180906_1700_II-01_CHIR2-4-6-8-10-12_24H-30H-36H-42H-48H_Tiles109_Z3_Time_5x2_CD09'
    # rename_CZI_IEXP(path, '2018-09-06~II-01_CD09')
    # path = r'X:\20180908_1600_II-02_CHIR2-4-6-8-10-12_24H-30H-36H-42H-48H_Tiles109_Z3_Time_5x2_CD09'
    # rename_CZI_IEXP(path, '2018-09-08~II-02_CD09')
    # path = r'X:\20180909_1600_III-01_CHIR2-4-6-8-10-12_24H-30H-36H-42H-48H_Tiles109_Z3_Time_5x2_CD09'
    # rename_CZI_IEXP(path, '2018-09-09~III-01_CD09')
    # path = r'X:\20180910_1900_III-02_CHIR2-4-6-8-10-12_24H-30H-36H-42H-48H_Tiles109_Z3_Time_5x2_CD09'
    # rename_CZI_IEXP(path, '2018-09-10~III-02_CD09')
    # path = r'X:\20180911_1600_III-03_CHIR2-4-6-8-10-12_24H-30H-36H-42H-48H_Tiles109_Z3_Time_5x2_CD09'
    # rename_CZI_IEXP(path, '2018-09-11~III-03_CD09')
    # make_movies(r'C:\Users\Kitty\Desktop\cd13\SSS_30%\S1', r'C:\Users\Kitty\Desktop\cd13\SSS_30%\S1\s.avi', 1000, 1000,
    #             fps=3, color=True)
    # research_stitching_image(r'J:\PROCESSING\CD13', 1)
    # extract_sp_image(r'J:\PROCESSING\CD13', repetition=False)
    # pca_analysis(r'C:\Users\Kitty\Desktop\CD13', r'C:\Users\Kitty\Desktop\CD13\Analysis', MAX_mle=3, pca_save=True,
    #              draw=False, draw_save=True, D=2, shape='-', do_all_pca=True)
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # draw_dispersed_pca(main_path, draw_folder='Figure_Merge_Dispersed_Circle', draw=False, well=96, D=2, shape='-',
    #                    x_min=-75, x_max=75, y_min=None, y_max=75, text=False)
    # merge_all_well_features(main_path, features_path)
    # do_pca(main_path, r'C:\Users\Kitty\Desktop\CD13\All_FEATURES.csv', 'PCA.csv', shape='.')
    # do_draw(main_path, input_csv_path, 'test.png', draw=False, D=2, shape='.', x_min=None, x_max=None, y_min=None,
    #         y_max=None)
    # do_draw_3(main_path, input_csv_path, 'colorful_PCA_10.png', draw=False, D=2, shape='.', x_min=None, x_max=None,
    #           y_min=None, y_max=None, IF_file='IF_Result_human.csv')
    # output_fold = 'Figure_dot_flow'
    # do_draw_dot_flow(main_path, input_csv_path, output_fold, IF_file, x_min=-75, x_max=75, y_min=None, y_max=75)
    # make_movies(pic_path, output_file, 1280, 1024, fps=3, color=True)
    # do_draw_dot_flow_3(main_path, input_csv_path, 'Figure_dot_flow_3_text_very_small', IF_file,
    #                    fig_size=(12.80 * 4, 10.24 * 4), x_min=-75, x_max=75, y_min=None, y_max=75, text=True)
    # pic_path = r'C:\Users\Kitty\Desktop\CD13\Figure_dot_flow_3_text_small'
    # output_file = r'C:\Users\Kitty\Desktop\CD13\ALL_Figure_dot_flow_3_text_small.avi'
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\PCA.csv'
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\Time_Series_PCA.csv'
    # do_draw_dot_flow_CHIR(main_path, input_csv_vertical, input_csv_horizontal, 'Figure_dot_flow_CHIR',
    #                       fig_size=(12.80, 10.24), x_min=-75, x_max=75, y_min=None, y_max=75, text=False)
    # do_draw_dot_flow_CHIR(main_path, input_csv_vertical, input_csv_horizontal, 'Figure_dot_flow_CHIR_text_small',
    #                       fig_size=(12.80*2, 10.24*2),  x_min=-75, x_max=75, y_min=None, y_max=75, text=True)
    # do_draw_dot_flow_CHIR(main_path, input_csv_vertical, input_csv_horizontal, 'Figure_dot_flow_CHIR_text_very_small',
    #                       fig_size=(12.80 * 4, 10.24 * 4), x_min=-75, x_max=75, y_min=None, y_max=75, text=True)
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # pic_path = r'C:\Users\Kitty\Desktop\CD13\Figure_dot_flow_CHIR'
    # output_file = r'C:\Users\Kitty\Desktop\CD13\Figure_dot_flow_CHIR.avi'
    # make_movies(pic_path, output_file, 1280, 1024, fps=3, color=True)
    # pic_path = r'C:\Users\Kitty\Desktop\CD13\Figure_dot_flow_CHIR_text_small'
    # output_file = r'C:\Users\Kitty\Desktop\CD13\Figure_dot_flow_CHIR_text_small.avi'
    # make_movies(pic_path, output_file, 1280 * 2, 1024 * 2, fps=3, color=True)
    # output_fold = 'test'
    # time_point = 1
    # do_draw_SP_time_point(main_path, input_csv_vertical, input_csv_horizontal, IF_file, output_fold, time_point,
    #                       fig_size=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None, text=False)
    # output_name = 'test.csv'
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\PCA.csv'
    # output_csv = r'PCA.csv'
    # do_pca(main_path, input_csv_path, output_csv, MAX_mle=3, draw_save=False, draw=False, D=2, shape='-', x_min=None,
    #        x_max=None, y_min=None, y_max=None)
    # make_movies(pic_path, output_file, 1280 * 2, 1024 * 2, fps=3, color=True)
    # make_movies(r'C:\Users\Kitty\Desktop\CD16\Whole_SSS_320', r'C:\Users\Kitty\Desktop\CD16\SSS.avi', 3840, 2560)
    # make_whole_96well_result(r'C:\Users\Kitty\Desktop\result_CD13_20%', 0.2)
    # print(make_movies(r'C:\Users\Kitty\Desktop\s04', r'C:\Users\Kitty\Desktop\1000.avi', 1000, 1000))
    # main_path = r'C:\Users\Kitty\Desktop\test'
    # path = r'C:\Users\Kitty\Desktop\test\2018-11-28\IPS_CD13'
    # matrix = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    # stitching_CZI_AE(main_path, path, 1, 1, 1, 2, 1, matrix, 1, 0.1)
    # path = r'C:\Users\Kitty\Desktop\20180903_1800_I-01_CHIR2-4-6-8-10-12_24H-30H-36H-42H-48H_Tiles109_Z3_Time_5x2_CD09'
    # rename_CZI_IEXP(path, '2018-09-03~I-01_CD09')
    # main_path = r'C:\Users\Kitty\Desktop\test'
    # path = r'C:\Users\Kitty\Desktop\test\2018-11-28\IPS_CD13'
    # matrix = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    # stitching_CZI(main_path, path, 1, 1, 1, 2, 1, matrix, 1, 0.1)
    # main_path = r'H:\CD09'
    # path = r'H:\CD09\Exported\2018-09-03~I-1_CD09'
    # matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                    [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                    [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                    [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]])
    # stitching_CZI(main_path, path, 1, 1, 1, 2, 1, matrix, 1, 0.1)
    # matrix = np.array(
    #     [[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    # stitching_CZI(main_path, path, 1, 1, 2, 2, 1, matrix, 1, 0.1)
    # stitching_CZI(main_path, path, 1, 1, 3, 2, 1, matrix, 1, 0.1)
    # stitching_CZI(main_path, path, 1, 1, 4, 2, 1, matrix, 1, 0.1)
    # stitching_CZI(main_path, path, 1, 1, 5, 2, 1, matrix, 1, 0.1)
    # stitching_CZI(main_path, path, 1, 1, 6, 2, 1, matrix, 1, 0.1)
    # print('now merge_all_well_features')
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # features_path = r'C:\Users\Kitty\Desktop\CD13\Features'
    # merge_all_well_features(main_path, features_path, output_name='All_FEATURES_new.csv')
    # print('now do_manifold')
    # time.sleep(10)
    # features_csv = r'C:\Users\Kitty\Desktop\CD13\All_FEATURES_new.csv'
    # output_folder = r'C:\Users\Kitty\Desktop\CD13\MainFold_ORB' # range(384,448)
    # output_folder = r'C:\Users\Kitty\Desktop\CD13\MainFold_SUR' # range(256,384)
    # output_folder = r'C:\Users\Kitty\Desktop\CD13\MainFold_All' # range(0,448)
    # do_manifold(main_path, features_csv, range(0,448), output_folder, n_neighbors=10, n_components=3)
