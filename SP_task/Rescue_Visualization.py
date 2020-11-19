import os
import time
import shutil
import random
from collections import Iterable
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.image as pltimg
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Lib_Class import ImageData


def rescue_v(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-48h', figsize=(12.80, 10.24),
             fontsize=18, sep_flag=False):
    # for Rescue Visualization, draw scatter !
    print('>>> ! Rescue Visualization!')

    # ---the pandas display settings---
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.width', None)

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(os.path.join(main_path, csv_file)):
        print('!ERROR! The All_DATA_PCA.csv does not existed!')
        return False

    # ,CHIR_0h-24h,CHIR_24h-48h,efficiency
    input_DF = pd.read_csv(os.path.join(main_path, csv_file), header=0, index_col=0)
    # input_DF_GB = input_DF.groupby(input_DF[['CHIR_0h-24h','CHIR_24h-48h']]).mean()
    input_DF_GB = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).mean()
    # print(input_DF_GB)
    # print(type(input_DF_GB))

    if sep_flag:
        input = input_DF.values
        shift = [-0.1, 0.1] * int(input.shape[0] / 2)
        input[:, 0] += shift
        # fig = plt.figure(figsize=(12.80, 10.24))
        fig = plt.figure(figsize=figsize)
        c = input[:, 2]
        plt.scatter(input[:, 0], input[:, 1], c=c)
        plt.title(csv_file, fontsize=fontsize)
        plt.xlabel('Original ' + original_str, fontsize=fontsize)
        plt.ylabel('Rescue ' + rescue_str, fontsize=fontsize)
        # plt.legend(loc='upper right')
        fig.savefig(os.path.join(main_path, csv_file + '.png'))
        # plt.show()
        plt.close()

    fig = plt.figure(figsize=figsize)
    c = input_DF_GB['efficiency'].values
    # print(c)
    tuple_list = list(input_DF_GB.index.values)
    loc_x = [0] * len(tuple_list)
    loc_y = [0] * len(tuple_list)
    for i in range(len(tuple_list)):
        loc_x[i] = list(tuple_list[i])[0]
        loc_y[i] = list(tuple_list[i])[1]
    # cm = plt.cm.get_cmap('RdYlBu')
    font_user = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': fontsize,
                 }
    s = [fontsize * 33 for i in range(len(c))]
    sc = plt.scatter(loc_x, loc_y, c=c, s=s)  # , cmap=cm)
    # v = np.linspace(0, 0.2, 1, endpoint=True)
    v = [0, 20, 40, 60, 80, 100]
    cb = plt.colorbar(ticks=v)
    # cb = plt.colorbar(sc)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('cTnT %', fontdict=font_user)
    # plt.title(csv_file, font_user)
    plt.tick_params(labelsize=fontsize)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14])
    plt.yticks([0, 2, 4, 6, 8, 10, 12, 14])
    plt.xlabel('Original ' + original_str + '  (uM)', font_user)
    plt.ylabel('Rescue ' + rescue_str + '  (uM)', font_user)
    # plt.legend(loc='upper right')
    fig.savefig(os.path.join(main_path, csv_file + '.png'))
    # plt.show()
    plt.close()

    return True


def concentration_v(main_path, csv_file, chir_str='chir', time_str='time', figsize=(12.80, 10.24),
             fontsize=18):
    # for Rescue Visualization, draw scatter !
    print('>>> ! concentration duration Visualization!')

    # ---the pandas display settings---
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.width', None)

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(os.path.join(main_path, csv_file)):
        print('!ERROR! The All_DATA_PCA.csv does not existed!')
        return False

    # ,CHIR_0h-24h,CHIR_24h-48h,efficiency
    input_DF = pd.read_csv(os.path.join(main_path, csv_file), header=0, index_col=0)
    # print(input_DF)
    # input_DF_GB = input_DF.groupby(input_DF[['CHIR_0h-24h','CHIR_24h-48h']]).mean()
    input_DF_GB = input_DF.groupby([input_DF[chir_str], input_DF[time_str]]).mean()
    # print(input_DF_GB)
    # print(type(input_DF_GB))

    fig = plt.figure(figsize=figsize)
    c = input_DF_GB['ef'].values
    # print(c)
    tuple_list = list(input_DF_GB.index.values)
    loc_x = [0] * len(tuple_list)
    loc_y = [0] * len(tuple_list)
    for i in range(len(tuple_list)):
        loc_x[i] = list(tuple_list[i])[0]
        loc_y[i] = list(tuple_list[i])[1]
    # cm = plt.cm.get_cmap('RdYlBu')
    font_user = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': fontsize,
                 }
    s = [fontsize * 33 for i in range(len(c))]
    sc = plt.scatter(loc_x, loc_y, c=c, s=s)  # , cmap=cm)
    # v = np.linspace(0, 0.2, 1, endpoint=True)
    v = [0, 20, 40, 60, 80, 100]
    cb = plt.colorbar(ticks=v)
    # cb = plt.colorbar(sc)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('cTnT %', fontdict=font_user)
    # plt.title(csv_file, font_user)
    plt.tick_params(labelsize=fontsize)
    plt.xticks([2, 4, 6, 8, 10, 12, 14])
    plt.yticks([24,36,48,60,72])
    plt.xlabel('CHIR Concentration (uM)', font_user)
    plt.ylabel('Duration (h)', font_user)
    # plt.legend(loc='upper right')
    fig.savefig(os.path.join(main_path, csv_file + '.png'))
    # plt.show()
    plt.close()

    return True

if __name__ == '__main__':
    main_path = r'S:\DC_f20191114\ZhaoLab_Current\ZhaoLab_Image\20190425组会_rescue\20190424_r'
    # csv_file = 'rescue-3.csv'
    #
    # rescue_v(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', figsize=(12.80, 10.24),
    #          fontsize=33, sep_flag=False)
    #
    # csv_file = 'rescue-4.csv'
    # rescue_v(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-32h', figsize=(12.80, 10.24),
    #          fontsize=33, sep_flag=False)

    csv_file = 'poster.csv'
    concentration_v(main_path, csv_file, chir_str='chir', time_str='time', figsize=(12.80, 10.24),
                    fontsize=33)
