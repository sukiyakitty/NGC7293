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
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.decomposition import PCA


# from Lib_Class import ImageData


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def rescue_V(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', efficiency_str='efficiency',
             xtcks=[0, 2, 4, 6, 8, 10, 12, 14], ytcks=[0, 2, 4, 6, 8, 10, 12, 14], figsize=(19.20, 16.80),
             fontsize=66, sep_flag=False):
    # for Rescue Visualization, draw scatter !
    # plot dot with fixed xtcks & ytcks, and draw all tagging

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
    c = input_DF_GB[efficiency_str].values
    c = 100 * normalization(c)
    # print(c)
    tuple_list = list(input_DF_GB.index.values)
    loc_x = [0] * len(tuple_list)
    loc_y = [0] * len(tuple_list)
    for i in range(len(tuple_list)):
        loc_x[i] = list(tuple_list[i])[0]
        loc_y[i] = list(tuple_list[i])[1]
    # cm = plt.cm.get_cmap('RdYlBu')
    font_user = {'family': 'Arial',
                 'weight': 'normal',
                 'size': fontsize,
                 }
    colors = [(0, 0, 0), (0.126453, 0.570633, 0.429841, 1.0), (0.281477, 0.755203, 0.362552, 1.0),
              (0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0),
              (0.993248, 0.906157, 0.143936, 1.0)]
    my_cmp = LinearSegmentedColormap.from_list("mycmap", colors)
    s = [fontsize * 33 for i in range(len(c))]
    sc = plt.scatter(loc_x, loc_y, c=c, s=s, cmap=my_cmp)
    # v = np.linspace(0, 0.2, 1, endpoint=True)
    v = [0, 20, 40, 60, 80, 100]
    cb = plt.colorbar(ticks=v)
    # cb = plt.colorbar(sc)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('cTnT %', fontdict=font_user)
    # plt.title(csv_file, font_user)
    plt.tick_params(labelsize=fontsize)
    # plt.xticks([0, 2, 4, 6, 8, 10, 12, 14])
    # plt.yticks([0, 2, 4, 6, 8, 10, 12, 14])
    plt.xticks(xtcks)
    plt.yticks(ytcks)
    plt.xlabel('Original ' + original_str + '  (uM)', font_user)
    plt.ylabel('Rescue ' + rescue_str + '  (uM)', font_user)
    # plt.legend(loc='upper right')
    fig.savefig(os.path.join(main_path, csv_file[:-4] + '.png'))
    # plt.show()
    plt.close()

    return True


def rescue_V2(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', efficiency_str='efficiency',
              xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66, sep_flag=False):
    # for Rescue Visualization, draw scatter !
    # plot dot with auto xtcks & ytcks, and draw all tagging

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
    if xtcks is None:
        xtcks = input_DF.groupby(input_DF[original_str]).count().index.values.tolist()
    if ytcks is None:
        ytcks = input_DF.groupby(input_DF[rescue_str]).count().index.values.tolist()
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
        fig.savefig(os.path.join(main_path, csv_file + '_V2.png'))
        # plt.show()
        plt.close()

    fig = plt.figure(figsize=figsize)
    c = input_DF_GB[efficiency_str].values
    c = 100 * normalization(c)
    # print(c)
    tuple_list = list(input_DF_GB.index.values)
    loc_x = [0] * len(tuple_list)
    loc_y = [0] * len(tuple_list)
    for i in range(len(tuple_list)):
        loc_x[i] = list(tuple_list[i])[0]
        loc_y[i] = list(tuple_list[i])[1]
    # cm = plt.cm.get_cmap('RdYlBu')
    font_user = {'family': 'Arial',
                 'weight': 'normal',
                 'size': fontsize,
                 }
    colors = [(0, 0, 0), (0.126453, 0.570633, 0.429841, 1.0), (0.281477, 0.755203, 0.362552, 1.0),
              (0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0),
              (0.993248, 0.906157, 0.143936, 1.0)]
    my_cmp = LinearSegmentedColormap.from_list("mycmap", colors)
    s = [fontsize * 33 for i in range(len(c))]
    sc = plt.scatter(loc_x, loc_y, c=c, s=s, cmap=my_cmp)
    # v = np.linspace(0, 0.2, 1, endpoint=True)
    v = [0, 20, 40, 60, 80, 100]
    cb = plt.colorbar(ticks=v)
    # cb = plt.colorbar(sc)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('cTnT %', fontdict=font_user)
    plt.title(csv_file, font_user)
    plt.tick_params(labelsize=fontsize)
    # plt.xticks([0, 2, 4, 6, 8, 10, 12, 14])
    # plt.yticks([0, 2, 4, 6, 8, 10, 12, 14])
    plt.xticks(xtcks)
    plt.yticks(ytcks)
    plt.xlabel('Original ' + original_str + '  (uM)', font_user)
    plt.ylabel('Rescue ' + rescue_str + '  (uM)', font_user)
    # plt.legend(loc='upper right')
    fig.savefig(os.path.join(main_path, csv_file[:-4] + '_V2.png'))
    # plt.show()
    plt.close()

    return True


def rescue_V3(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', efficiency_str='efficiency',
              xtcks=None, ytcks=None, figsize=(19.20, 16.80 / 2), fontsize=32):
    # for Rescue Visualization, bar groups !
    # draw bar groups with auto xtcks & ytcks, and draw legend

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
    if xtcks is None:
        xtcks = input_DF.groupby(input_DF[original_str]).count().index.values.tolist()
    if ytcks is None:
        ytcks = input_DF.groupby(input_DF[rescue_str]).count().index.values.tolist()
    input_DF_GB_mean = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).mean()
    input_DF_GB_std = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).var()
    # print(input_DF_GB)
    # print(type(input_DF_GB))
    # fig = plt.figure(figsize=figsize)

    font_user = {'family': 'Arial',
                 'weight': 'normal',
                 'size': fontsize,
                 }
    colors = [(0, 0, 0), (0.126453, 0.570633, 0.429841, 1.0), (0.281477, 0.755203, 0.362552, 1.0),
              (0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0),
              (0.993248, 0.906157, 0.143936, 1.0)]

    fig, ax = plt.subplots(figsize=figsize)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.8, chartBox.height])
    index = np.arange(len(xtcks))
    bar_width = 1 / (len(ytcks) + len(ytcks) / 2)
    opacity = 0.618
    # error_config = {'ecolor': '0.01'}
    for y in np.arange(len(ytcks)):
        rects = ax.bar(index + y * bar_width,
                       input_DF_GB_mean[efficiency_str][:, ytcks[y]],
                       bar_width, color=colors[y],
                       alpha=opacity,
                       yerr=2 * input_DF_GB_std[efficiency_str][:, ytcks[y]],
                       label='CHIR-' + str(ytcks[y]))

    # yerr = input_DF_GB_std[efficiency_str][:, ytcks[y]],
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    # ax.set_ylabel('Scores')
    # ax.set_title('rescue')
    v_ = [0, 0.2, 0.4, 0.6, 0.8, 1]
    v = ['0%', '20%', '40%', '60%', '80%', '100%']
    plt.yticks(v_, v)
    # plt.yticks(ytcks)
    ax.set_xticks(index + 0.5 * bar_width * (len(ytcks) - 1))
    ax.set_xticklabels(('CHIR-' + str(i) for i in xtcks))
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize,
                    title='Resecue ' + rescue_str.split('CHIR ')[-1])
    lgd.get_title().set_fontsize(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    fig.savefig(os.path.join(main_path, csv_file[:-4] + '_V3.png'))
    # plt.show()
    plt.close()

    return True


def rescue_V4(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', efficiency_str='efficiency',
              xtcks=None, ytcks=None, figsize=(19.20, 16.80 / 2), fontsize=32):
    # for Rescue Visualization, bar groups !
    # draw bar groups with auto xtcks & ytcks, and draw NO legend

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
    if xtcks is None:
        xtcks = input_DF.groupby(input_DF[original_str]).count().index.values.tolist()
    if ytcks is None:
        ytcks = input_DF.groupby(input_DF[rescue_str]).count().index.values.tolist()
    input_DF_GB_mean = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).mean()
    input_DF_GB_std = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).var()
    # print(input_DF_GB)
    # print(type(input_DF_GB))
    # fig = plt.figure(figsize=figsize)

    font_user = {'family': 'Arial',
                 'weight': 'normal',
                 'size': fontsize,
                 }
    colors = [(0, 0, 0), (0.126453, 0.570633, 0.429841, 1.0), (0.281477, 0.755203, 0.362552, 1.0),
              (0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0),
              (0.993248, 0.906157, 0.143936, 1.0)]

    fig, ax = plt.subplots(figsize=figsize)
    # chartBox = ax.get_position()
    # ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.8, chartBox.height])
    index = np.arange(len(xtcks))
    bar_width = 1 / (len(ytcks) + len(ytcks) / 2)
    opacity = 0.918
    # error_config = {'ecolor': '0.01'}
    for y in np.arange(len(ytcks)):
        rects = ax.bar(index + y * bar_width,
                       input_DF_GB_mean[efficiency_str][:, ytcks[y]],
                       bar_width, color=colors[y],
                       alpha=opacity,
                       yerr=2 * input_DF_GB_std[efficiency_str][:, ytcks[y]],
                       label='CHIR-' + str(ytcks[y]))

    # yerr = input_DF_GB_std[efficiency_str][:, ytcks[y]],
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    # ax.set_ylabel('Scores')
    # ax.set_title('rescue')
    v_ = [0, 0.2, 0.4, 0.6, 0.8, 1]
    v = ['0%', '20%', '40%', '60%', '80%', '100%']
    plt.yticks(v_, v)
    # plt.yticks(ytcks)
    ax.set_xticks(index + 0.5 * bar_width * (len(ytcks) - 1))
    ax.set_xticklabels(('CHIR-' + str(i) for i in xtcks))
    # lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize,
    #           title='Resecue ' + rescue_str.split('CHIR ')[-1])
    # lgd.get_title().set_fontsize(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    fig.savefig(os.path.join(main_path, csv_file[:-4] + '_V4.png'))
    # plt.show()
    plt.close()

    return True


def rescue_V5(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', efficiency_str='efficiency',
              xtcks=None, ytcks=None, figsize=(19.20, 16.80 / 2), fontsize=32):
    # for Rescue Visualization, bar groups !
    # draw bar groups with auto xtcks & ytcks, and draw NO legend, NO xticklabels

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

    input_DF = pd.read_csv(os.path.join(main_path, csv_file), header=0, index_col=0)
    if xtcks is None:
        xtcks = input_DF.groupby(input_DF[original_str]).count().index.values.tolist()
    if ytcks is None:
        ytcks = input_DF.groupby(input_DF[rescue_str]).count().index.values.tolist()
    input_DF_GB_mean = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).mean()
    input_DF_GB_std = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).var()

    font_user = {'family': 'Arial',
                 'weight': 'normal',
                 'size': fontsize,
                 }
    colors = [(0, 0, 0), (0.126453, 0.570633, 0.429841, 1.0), (0.281477, 0.755203, 0.362552, 1.0),
              (0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0),
              (0.993248, 0.906157, 0.143936, 1.0)]

    fig, ax = plt.subplots(figsize=figsize)
    index = np.arange(len(xtcks))
    bar_width = 1 / (len(ytcks) + len(ytcks) / 2)
    opacity = 0.918
    for y in np.arange(len(ytcks)):
        rects = ax.bar(index + y * bar_width,
                       input_DF_GB_mean[efficiency_str][:, ytcks[y]],
                       bar_width, color=colors[y],
                       alpha=opacity,
                       yerr=2 * input_DF_GB_std[efficiency_str][:, ytcks[y]],
                       label='CHIR-' + str(ytcks[y]))

    # yerr = input_DF_GB_std[efficiency_str][:, ytcks[y]],
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    # ax.set_ylabel('Scores')
    # ax.set_title('rescue')
    v_ = [0, 0.2, 0.4, 0.6, 0.8, 1]
    v = ['0%', '20%', '40%', '60%', '80%', '100%']
    plt.yticks(v_, v)
    # plt.yticks(ytcks)
    ax.set_xticks(index + 0.5 * bar_width * (len(ytcks) - 1))
    ax.set_xticklabels(('' for i in xtcks))
    # lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize,
    #           title='Resecue ' + rescue_str.split('CHIR ')[-1])
    # lgd.get_title().set_fontsize(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    fig.savefig(os.path.join(main_path, csv_file[:-4] + '_V5.png'))
    # plt.show()
    plt.close()

    return True


def rescue_V6(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', efficiency_str='efficiency',
              xtcks=None, ytcks=None, figsize=(19.20, 16.80 / 2), fontsize=32):
    # for Rescue Visualization, bar groups !
    # draw bar groups with auto xtcks & ytcks, and draw NO legend, NO xticklabels

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

    input_DF = pd.read_csv(os.path.join(main_path, csv_file), header=0, index_col=0)
    if xtcks is None:
        xtcks = input_DF.groupby(input_DF[original_str]).count().index.values.tolist()
    if ytcks is None:
        ytcks = input_DF.groupby(input_DF[rescue_str]).count().index.values.tolist()
    input_DF_GB_mean = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).mean()
    input_DF_GB_std = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).var()

    font_user = {'family': 'Arial',
                 'weight': 'normal',
                 'size': fontsize,
                 }
    colors = [(0, 0, 0), (0.126453, 0.570633, 0.429841, 1.0), (0.281477, 0.755203, 0.362552, 1.0),
              (0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0),
              (0.993248, 0.906157, 0.143936, 1.0)]

    fig, ax = plt.subplots(figsize=figsize)
    index = np.arange(len(xtcks))
    bar_width = 1 / (len(ytcks) + len(ytcks) / 2)
    opacity = 0.618
    for y in np.arange(len(ytcks)):
        rects = ax.bar(index + y * bar_width,
                       input_DF_GB_mean[efficiency_str][:, ytcks[y]],
                       bar_width,
                       alpha=opacity,
                       # yerr=2 * input_DF_GB_std[efficiency_str][:, ytcks[y]],
                       label='CHIR-' + str(ytcks[y]))

    # yerr = input_DF_GB_std[efficiency_str][:, ytcks[y]],
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    # ax.set_ylabel('Scores')
    # ax.set_title('rescue')
    v_ = [0, 0.2, 0.4, 0.6, 0.8, 1]
    v = ['0%', '20%', '40%', '60%', '80%', '100%']
    plt.yticks(v_, v)
    # plt.yticks(ytcks)
    ax.set_xticks(index + 0.5 * bar_width * (len(ytcks) - 1))
    ax.set_xticklabels(('' for i in xtcks))
    # lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize,
    #           title='Resecue ' + rescue_str.split('CHIR ')[-1])
    # lgd.get_title().set_fontsize(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    fig.savefig(os.path.join(main_path, csv_file[:-4] + '_V6.png'))
    # plt.show()
    plt.close()

    return True


def rescue_V7(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', efficiency_str='efficiency',
              xtcks=None, ytcks=None, figsize=(19.20, 16.80 / 2), fontsize=32):
    # for Rescue Visualization, bar groups !
    # draw bar groups with auto xtcks & ytcks, and draw NO legend, NO xticklabels

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

    input_DF = pd.read_csv(os.path.join(main_path, csv_file), header=0, index_col=0)
    if xtcks is None:
        xtcks = input_DF.groupby(input_DF[original_str]).count().index.values.tolist()
    if ytcks is None:
        ytcks = input_DF.groupby(input_DF[rescue_str]).count().index.values.tolist()
    input_DF_GB_mean = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).mean()
    input_DF_GB_std = input_DF.groupby([input_DF[original_str], input_DF[rescue_str]]).var()

    font_user = {'family': 'Arial',
                 'weight': 'normal',
                 'size': fontsize,
                 }
    colors_list = [(0.076615, 0.626376, 0.086377, 1.0), (0.883248, 0.866157, 0.173936, 1.0)]
    my_cmp = LinearSegmentedColormap.from_list("mycmap", colors_list, N=len(ytcks))
    colors_list = [[my_cmp(i)[0], my_cmp(i)[1], my_cmp(i)[2]] for i in range(my_cmp.N)]

    fig, ax = plt.subplots(figsize=figsize)
    index = np.arange(len(xtcks))
    bar_width = 1 / (len(ytcks) + len(ytcks) / 2)
    opacity = 0.818
    for y in np.arange(len(ytcks)):
        rects = ax.bar(index + y * bar_width,
                       input_DF_GB_mean[efficiency_str][:, ytcks[y]],
                       bar_width, color=colors_list[y],
                       alpha=opacity,
                       # yerr=2 * input_DF_GB_std[efficiency_str][:, ytcks[y]],
                       label='CHIR-' + str(ytcks[y]))

    # yerr = input_DF_GB_std[efficiency_str][:, ytcks[y]],
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    # ax.set_ylabel('Scores')
    # ax.set_title('rescue')
    v_ = [0, 0.2, 0.4, 0.6, 0.8, 1]
    v = ['0%', '20%', '40%', '60%', '80%', '100%']
    plt.yticks(v_, v)
    # plt.yticks(ytcks)
    ax.set_xticks(index + 0.5 * bar_width * (len(ytcks) - 1))
    ax.set_xticklabels(('' for i in xtcks))
    # lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize,
    #           title='Resecue ' + rescue_str.split('CHIR ')[-1])
    # lgd.get_title().set_fontsize(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    # ax.set_xlabel('Initial ' + original_str.split('CHIR ')[-1])
    fig.savefig(os.path.join(main_path, csv_file[:-4] + '_V7.png'))
    # plt.show()
    plt.close()

    return True


def concentration_v(main_path, csv_file, chir_str='chir', time_str='time', efficiency_str='efficiency',
                    xtcks=[2, 4, 6, 8, 10, 12, 14], ytcks=[24, 36, 48, 60, 72], figsize=(19.20, 16.80), fontsize=66):
    # for Rescue Visualization, draw scatter !
    # plot dot with fixed xtcks & ytcks, and draw all tagging

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
    c = input_DF_GB[efficiency_str].values
    c = 100 * normalization(c)
    # print(c)
    tuple_list = list(input_DF_GB.index.values)
    loc_x = [0] * len(tuple_list)
    loc_y = [0] * len(tuple_list)
    for i in range(len(tuple_list)):
        loc_x[i] = list(tuple_list[i])[0]
        loc_y[i] = list(tuple_list[i])[1]
    # cm = plt.cm.get_cmap('RdYlBu')
    font_user = {'family': 'Arial',
                 'weight': 'normal',
                 'size': fontsize,
                 }

    # viridis_big = cm.get_cmap('viridis', 500)
    # my_cmp = ListedColormap(viridis_big(np.linspace(0.8, 1, 100)))
    # N = 256
    # vals = np.ones((N, 4))
    # vals[:, 0] = np.linspace(0, 1, N)
    # vals[:, 1] = np.linspace(1, 1, N)
    # vals[:, 2] = np.linspace(0, 0, N)
    # my_cmp = ListedColormap(vals)
    # colors = [(0,0,0),(0.126453, 0.570633, 0.549841, 1.0),(0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0), (0.993248, 0.906157, 0.143936, 1.0)]
    # colors = [(0,0,0),(0.126453, 0.570633, 0.549841, 1.0),(0.14021, 0.665859, 0.513427, 1.0),(0.281477, 0.755203, 0.432552, 1.0),(0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0), (0.993248, 0.906157, 0.143936, 1.0)]
    colors = [(0, 0, 0), (0.126453, 0.570633, 0.429841, 1.0), (0.281477, 0.755203, 0.362552, 1.0),
              (0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0),
              (0.993248, 0.906157, 0.143936, 1.0)]
    my_cmp = LinearSegmentedColormap.from_list("mycmap", colors)
    s = [fontsize * 33 for i in range(len(c))]
    sc = plt.scatter(loc_x, loc_y, c=c, s=s, cmap=my_cmp)
    # v = np.linspace(0, 0.2, 1, endpoint=True)
    v = [0, 20, 40, 60, 80, 100]
    # v = [0, 0.2, 0.4, 0.6, 0.8, 1]
    cb = plt.colorbar(ticks=v)
    # cb = plt.colorbar(sc)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('cTnT %', fontdict=font_user)
    # plt.title(csv_file, font_user)
    plt.tick_params(labelsize=fontsize)
    # plt.xticks([2, 4, 6, 8, 10, 12, 14])
    # plt.xticks([2, 4, 6, 8, 10, 12])
    # plt.yticks([24, 36, 48, 60, 72])
    # plt.yticks([24, 36, 48])
    plt.xticks(xtcks)
    plt.yticks(ytcks)
    plt.xlabel('CHIR Concentration (uM)', font_user)
    plt.ylabel('Duration (h)', font_user)
    # plt.legend(loc='upper right')
    fig.savefig(os.path.join(main_path, csv_file[:-4] + '.png'))
    # plt.show()
    plt.close()

    return True


def concentration_v2(main_path, csv_file, chir_str='chir', time_str='chir_hour', efficiency_str='IF_human',
                     xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66):
    # for Rescue Visualization, draw scatter !
    # plot dot with auto xtcks & ytcks, and draw NO tagging

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
        print('!ERROR! The csv does not existed!')
        return False

    # ,CHIR_0h-24h,CHIR_24h-48h,efficiency
    input_DF = pd.read_csv(os.path.join(main_path, csv_file), header=0, index_col=0)
    # print(input_DF)
    # input_DF_GB = input_DF.groupby(input_DF[['CHIR_0h-24h','CHIR_24h-48h']]).mean()
    if xtcks is None:
        xtcks = input_DF.groupby(input_DF[chir_str]).count().index.values.tolist()
    if ytcks is None:
        ytcks = input_DF.groupby(input_DF[time_str]).count().index.values.tolist()
    input_DF_GB = input_DF.groupby([input_DF[chir_str], input_DF[time_str]]).mean()
    # print(input_DF_GB)
    # print(type(input_DF_GB))

    fig = plt.figure(figsize=figsize)
    c = input_DF_GB[efficiency_str].values
    c = 100 * normalization(c)
    # print(c)
    tuple_list = list(input_DF_GB.index.values)
    loc_x = [0] * len(tuple_list)
    loc_y = [0] * len(tuple_list)
    for i in range(len(tuple_list)):
        loc_x[i] = list(tuple_list[i])[0]
        loc_y[i] = list(tuple_list[i])[1]
    # cm = plt.cm.get_cmap('RdYlBu')
    font_user = {'family': 'Arial',
                 'weight': 'normal',
                 'size': fontsize,
                 }

    # viridis_big = cm.get_cmap('viridis', 500)
    # my_cmp = ListedColormap(viridis_big(np.linspace(0.8, 1, 100)))
    # N = 256
    # vals = np.ones((N, 4))
    # vals[:, 0] = np.linspace(0, 1, N)
    # vals[:, 1] = np.linspace(1, 1, N)
    # vals[:, 2] = np.linspace(0, 0, N)
    # my_cmp = ListedColormap(vals)
    # colors = [(0,0,0),(0.126453, 0.570633, 0.549841, 1.0),(0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0), (0.993248, 0.906157, 0.143936, 1.0)]
    # colors = [(0,0,0),(0.126453, 0.570633, 0.549841, 1.0),(0.14021, 0.665859, 0.513427, 1.0),(0.281477, 0.755203, 0.432552, 1.0),(0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0), (0.993248, 0.906157, 0.143936, 1.0)]
    colors = [(0, 0, 0), (0.126453, 0.570633, 0.429841, 1.0), (0.281477, 0.755203, 0.362552, 1.0),
              (0.496615, 0.826376, 0.306377, 1.0), (0.762373, 0.876424, 0.137064, 1.0),
              (0.993248, 0.906157, 0.143936, 1.0)]
    my_cmp = LinearSegmentedColormap.from_list("mycmap", colors)
    s = [fontsize * 33 for i in range(len(c))]
    sc = plt.scatter(loc_x, loc_y, c=c, s=s, cmap=my_cmp)
    # v = np.linspace(0, 0.2, 1, endpoint=True)
    v = [0, 20, 40, 60, 80, 100]
    # v = [0, 0.2, 0.4, 0.6, 0.8, 1]
    cb = plt.colorbar(ticks=v)
    # cb = plt.colorbar(sc)
    cb.ax.tick_params(labelsize=fontsize)
    # cb.set_label('cTnT %', fontdict=font_user)
    # plt.title(csv_file, font_user)
    plt.tick_params(labelsize=fontsize)
    # plt.xticks([2, 4, 6, 8, 10, 12, 14])
    # plt.xticks([2, 4, 6, 8, 10, 12])
    # plt.yticks([24, 36, 48, 60, 72])
    # plt.yticks([24, 36, 48])
    plt.xticks(xtcks)
    plt.yticks(ytcks)
    # plt.xlabel('CHIR Concentration (uM)', font_user)
    # plt.ylabel('Duration (h)', font_user)
    # plt.legend(loc='upper right')
    fig.savefig(os.path.join(main_path, csv_file[:-4] + '.png'))
    # plt.show()
    plt.close()

    return True


def plot_examples(colormaps):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()


if __name__ == '__main__':
    main_path = r'D:\Orange\Study_ZhaoLab\ZhaoLab_Image\20201105_Rescue大总结+第一阶段Fractal大总结\Rescue\All_Rescue'
    csv_file = r'rescue-1.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-48h')
    csv_file = r'rescue-2.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-48h')
    csv_file = r'rescue-3.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-48h')
    csv_file = r'rescue-4.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-32h')

    csv_file = r'rescue-5__1.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-18h', rescue_str='CHIR_18h-30h')
    csv_file = r'rescue-5__2.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-18h', rescue_str='CHIR_18h-42h')
    csv_file = r'rescue-5__3.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-36h')
    csv_file = r'rescue-5__4.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-48h')
    csv_file = r'rescue-5__5.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-30h', rescue_str='CHIR_30h-42h')
    csv_file = r'rescue-5__6.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-36h', rescue_str='CHIR_36h-48h')

    csv_file = r'rescue-6__1.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-12h', rescue_str='CHIR_12h-24h')
    csv_file = r'rescue-6__2.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-12h', rescue_str='CHIR_12h-36h')
    csv_file = r'rescue-6__3.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-12h', rescue_str='CHIR_12h-48h')
    csv_file = r'rescue-6__4.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-12h', rescue_str='CHIR_12h-60h')
    csv_file = r'rescue-6__5.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-12h', rescue_str='CHIR_12h-72h')
    csv_file = r'rescue-6__6.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-36h')
    csv_file = r'rescue-6__7.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-48h')
    csv_file = r'rescue-6__8.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-60h')
    csv_file = r'rescue-6__9.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-24h', rescue_str='CHIR_24h-72h')
    csv_file = r'rescue-6__10.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-36h', rescue_str='CHIR_36h-48h')
    csv_file = r'rescue-6__11.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-36h', rescue_str='CHIR_36h-60h')
    csv_file = r'rescue-6__12.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-36h', rescue_str='CHIR_36h-72h')
    csv_file = r'rescue-6__13.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-48h', rescue_str='CHIR_48h-60h')
    csv_file = r'rescue-6__14.csv'
    rescue_V2(main_path, csv_file, original_str='CHIR_0h-48h', rescue_str='CHIR_48h-72h')


    # viridis = cm.get_cmap('viridis', 100)
    # print('viridis(1)', viridis(1))
    # print('viridis(0.95)', viridis(0.95))
    # print('viridis(0.9)', viridis(0.9))
    # print('viridis(0.8)', viridis(0.8))
    # print('viridis(0.7)', viridis(0.7))
    # print('viridis(0.6)', viridis(0.6))
    # print('viridis(0.5)', viridis(0.5))
    # print('viridis(0.4)', viridis(0.4))
    # print('viridis(0.3)', viridis(0.3))
    # print('viridis(0.2)', viridis(0.2))
    # print('viridis(0.1)', viridis(0.1))
    # print('viridis(0)', viridis(0))

    # (0.14021, 0.665859, 0.513427, 1.0)
    # (0.993248, 0.906157, 0.143936, 1.0)

    # cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
    # plot_examples([cmap])

    # viridis_big = cm.get_cmap('viridis', 512)
    # nmy_cmp = ListedColormap(viridis_big(np.linspace(0.5, 1, 256)))
    # plot_examples([viridis, nmy_cmp])

    # csv_file = 'rescue-3.csv'
    #
    # rescue_v(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', figsize=(12.80, 10.24),
    #          fontsize=33, sep_flag=False)
    #
    # csv_file = 'rescue-4.csv'
    # rescue_v(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-32h', figsize=(12.80, 10.24),
    #          fontsize=33, sep_flag=False)

    # main_path = r'C:\C137\USING\ZhaoLab\ZhaoLab_Image\20190425组会_rescue\20190424_r'
    # main_path = r'C:\C137\USING\ZhaoLab\ZhaoLab_Image\20201204_poster'
    #
    # csv_file = 'rescue-3.csv'
    # rescue_v(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h',
    #          xtcks=[0, 2, 4, 6, 8, 10, 12, 14], ytcks=[0, 2, 4, 6, 8, 10, 12, 14], figsize=(19.20, 16.80),
    #          fontsize=66, sep_flag=False)
    # csv_file = 'rescue-4.csv'
    # rescue_v(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-32h',
    #          xtcks=[0, 2, 4, 6, 8, 10, 12, 14], ytcks=[0, 2, 4, 6, 8, 10, 12, 14], figsize=(19.20, 16.80),
    #          fontsize=66, sep_flag=False)
    #
    # csv_file = 'poster_2019.csv'
    # concentration_v(main_path, csv_file, chir_str='chir', time_str='time', xtcks=[0, 2, 4, 6, 8, 10, 12, 14, 16],
    #                 ytcks=[12, 24, 36, 48, 60, 72, 84], figsize=(19.20, 16.80), fontsize=66)
    # csv_file = 'poster_2020.csv'
    # concentration_v(main_path, csv_file, chir_str='chir', time_str='time', xtcks=[0, 2, 4, 6, 8, 10, 12, 14],
    #                 ytcks=[12, 24, 36, 48, 60], figsize=(19.20, 16.80), fontsize=66)

    # main_path = r'D:\Green\Sub_Projects\ML_assists_hiPSC-CM\第一阶段Curve\20220406_chir_time'
    # csv_file = r'CD11.csv'
    # concentration_v2(main_path, csv_file, chir_str='chir', time_str='chir_hour', efficiency_str='IF_human',
    #                  xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66)
    # csv_file = r'CD13.csv'
    # concentration_v2(main_path, csv_file, chir_str='chir', time_str='chir_hour', efficiency_str='IF_human',
    #                  xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66)
    # csv_file = r'CD26.csv'
    # concentration_v2(main_path, csv_file, chir_str='chir', time_str='chir_hour', efficiency_str='IF_human',
    #                  xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66)
    # csv_file = r'CD27.csv'
    # concentration_v2(main_path, csv_file, chir_str='chir', time_str='chir_hour', efficiency_str='IF_human',
    #                  xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66)
    # csv_file = r'p1.csv'
    # concentration_v2(main_path, csv_file, chir_str='chir', time_str='time', efficiency_str='efficiency',
    #                  xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66)
    # csv_file = r'p2.csv'
    # concentration_v2(main_path, csv_file, chir_str='chir', time_str='time', efficiency_str='efficiency',
    #                  xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66)
    # csv_file = r'p3.csv'
    # concentration_v2(main_path, csv_file, chir_str='chir', time_str='time', efficiency_str='efficiency',
    #                  xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66)
    # csv_file = r'p4.csv'
    # concentration_v2(main_path, csv_file, chir_str='chir', time_str='time', efficiency_str='efficiency',
    #                  xtcks=None, ytcks=None, figsize=(19.20, 16.80), fontsize=66)

    # csv_file = 'rescue-3.csv'
    # rescue_v2(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', figsize=(19.20, 16.80),
    #           fontsize=66, sep_flag=False)
    # rescue_V2(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', figsize=(19.20, 16.80))
    # rescue_V7(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-48h', figsize=(19.20, 16.80 / 3))

    # csv_file = 'rescue-4.csv'
    # rescue_v2(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-32h', figsize=(19.20, 16.80),
    #           fontsize=66, sep_flag=False)
    # rescue_V2(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-32h', figsize=(19.20, 16.80))
    # rescue_V7(main_path, csv_file, original_str='CHIR 0h-24h', rescue_str='CHIR 24h-32h', figsize=(19.20, 16.80 / 3))
