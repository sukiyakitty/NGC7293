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
from Lib_Function import is_number
from Lib_Features import merge_specific_time_point_features, merge_all_well_features
from Lib_Manifold import do_pca, pca_vertical_to_horizontal


def draw_dispersed_pca(main_path, input_csv, draw_folder='PCAFigure_Merge_Dispersed', draw=False, well=96, D=2,
                       shape='-', x_min=None, x_max=None, y_min=None, y_max=None, text=False,
                       IF_file='IF_Result_human.csv'):
    # draw dispersed pca
    # input: main_path: main path;
    # input_csv: 'All_DATA_PCA.csv'
    # analysis_path: the path contained features .cvs files
    # MAX_mle = pca numbers
    # pca_save=True : save the pca result?
    # draw=False : do draw the pca visualization on screen? (NOTICE if draw_save=False, always do not draw)
    # draw_save=True : save the pca visualization result to .png
    # D=2 : pca visualization dimension
    # shape='-' : the matplotlib plot shape '-' is line ; '.' is dot
    # do_all_pca=True : combine all features, and do pca once for all
    # text=False : print time point number text on pca image, from 1 to n (reference to features.csv files)

    plt_circle = False
    circle_x = 20
    circle_y = 0
    circle_r = 10
    IF_result = None

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(os.path.join(main_path, input_csv)):
        print('!ERROR! The All_DATA_PCA.csv does not existed!')
        return False
    if os.path.exists(os.path.join(main_path, IF_file)):
        IF_result = pd.read_csv(os.path.join(main_path, IF_file), header=0, index_col=0)

    # draw_folder = 'Figure_Merge_Dispersed'
    if os.path.exists(os.path.join(main_path, draw_folder)):
        shutil.rmtree(os.path.join(main_path, draw_folder))
    os.makedirs(os.path.join(main_path, draw_folder))

    img_count_list = [0] * well
    pca_result_DF = pd.read_csv(os.path.join(main_path, input_csv), header=0, index_col=0)
    pca_result = pca_result_DF.values

    for x in pca_result_DF.index:
        img_count_list[int(x.split('~')[0].split('S')[1]) - 1] += 1

    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    i_index = 0
    for i in range(0, len(img_count_list)):
        i_range = range(i_index, i_index + img_count_list[i])
        i_index = i_index + img_count_list[i]

        fig = plt.figure(figsize=(12.80, 10.24))

        c = range(1, img_count_list[i] + 1)
        # c = pca_result[i_range, 2]
        color = np.random.rand(3)

        if D == 2:
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            if shape == '-':
                # plt.plot(pca_result[:, 0], pca_result[:, 1], color=color, marker='.', linestyle='-', label=ana_csv[:-4])
                plt.plot(pca_result[i_range, 0], pca_result[i_range, 1], color=color, linestyle='-',
                         label='S' + str(i + 1))
                plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c)
            elif shape == '.':
                plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c)
            else:
                plt.plot(pca_result[i_range, 0], pca_result[i_range, 1], color=color, marker=shape,
                         label='S' + str(i + 1))
            if text:
                for j in range(0, img_count_list[i]):
                    k = i_range[j]
                    plt.text(pca_result[k, 0], pca_result[k, 1], str(j + 1))
        elif D == 3:
            ax = plt.axes(projection='3d')
            ax.plot3D(pca_result[i_range, 0], pca_result[i_range, 1], pca_result[i_range, 2], shape,
                      label='S' + str(i + 1))
            # ax.scatter3D(pca_result[:,0],pca_result[:,1],pca_result[:,2])
        else:
            print('!ERROR! The D does not support!')
            return False

        if IF_result is not None:
            title_str = 'S' + str(i + 1) + ' IF Result: ' + str(IF_result['IF_intensity'].values[i])
        else:
            title_str = 'S' + str(i + 1) + '.png'

        if plt_circle:
            theta = np.linspace(0, 2 * np.pi, 210)
            x, y = np.cos(theta) * circle_r, np.sin(theta) * circle_r
            plt.plot(x + circle_x, y + circle_y, color='orangered', linewidth=2.0)
            x, y = np.cos(theta) * (circle_r + 5), np.sin(theta) * (circle_r + 5)
            plt.plot(x + circle_x, y + circle_y, color='orange', linewidth=2.0)
            x, y = np.cos(theta) * (circle_r + 10), np.sin(theta) * (circle_r + 10)
            plt.plot(x + circle_x, y + circle_y, color='yellow', linewidth=2.0)

        plt.title(title_str)
        plt.legend(loc='upper right')
        fig.savefig(os.path.join(main_path, draw_folder, 'S' + str(i + 1) + '.png'))
        if draw:
            plt.show()
        plt.close()
    return True


def do_draw_one_time_CD13(main_path, input_csv, IF_file, figsize=(12.80, 10.24), x_min=None, x_max=None, y_min=None,
                          y_max=None, text=False):
    # draw one time point pca picture
    # input:
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv):
        print('!ERROR! The input_csv_path does not existed!')
        return False
    if not os.path.exists(IF_file):
        print('!ERROR! The IF_file does not existed!')
        return False

    pca_result_DF = pd.read_csv(input_csv, header=0, index_col=0)
    IF_result_DF = pd.read_csv(os.path.join(main_path, IF_file), header=0, index_col=0)
    name = os.path.split(input_csv)[1].split('.csv')[0]

    if x_min is None:
        x_min = pca_result_DF.values[:, 0].min()
    if x_max is None:
        x_max = pca_result_DF.values[:, 0].max()
    if y_min is None:
        y_min = pca_result_DF.values[:, 1].min()
    if y_max is None:
        y_max = pca_result_DF.values[:, 1].max()

    IF_result_list = []
    for i in range(1, len(pca_result_DF) + 1):
        i_str = 'S' + str(i)
        if i_str in IF_result_DF.index:
            IF_result_list.append(IF_result_DF.loc[i_str, 'IF_intensity'])
        else:
            IF_result_list.append(0)

    # CD13 CHIR distribution
    dot_color_CHIR = []
    for i in range(0, len(pca_result_DF)):
        if i <= 23:  # chir 4
            dot_color_CHIR.append('navy')
        elif i <= 47:  # chir 6
            dot_color_CHIR.append('limegreen')
        elif i <= 71:  # chir 8
            dot_color_CHIR.append('gold')
        else:  # chir 10
            dot_color_CHIR.append('r')
    # CD13 CHIR distribution

    dot_color_IF = []
    for i in IF_result_list:
        if i <= 0.2:
            dot_color_IF.append('k')
        elif i <= 0.4:
            dot_color_IF.append('navy')
        elif i <= 0.6:
            dot_color_IF.append('gold')
        elif i <= 0.8:
            dot_color_IF.append('darkorange')
        else:
            dot_color_IF.append('r')

    fig = plt.figure(figsize=figsize)  # fig_size=(12.80, 10.24)
    plt.title(name + '_CHIR')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    for i_dot in range(0, len(pca_result_DF)):
        this_S_str = pca_result_DF.index[i_dot].split('~')[0]
        this_S_int = int(this_S_str.split('S')[1])
        plot_X = pca_result_DF.iloc[i_dot, 0]
        plot_Y = pca_result_DF.iloc[i_dot, 1]
        # plt.plot(plot_X, plot_Y, color=dot_color_CHIR[i_dot], linestyle='-')
        plt.scatter(plot_X, plot_Y, color=dot_color_CHIR[this_S_int - 1])
        if text:
            text_X = plot_X
            text_Y = plot_Y
            plt.text(text_X, text_Y, this_S_str)
    fig.savefig(
        os.path.join(main_path, name + '_CHIR.png'))
    plt.close()

    fig = plt.figure(figsize=figsize)  # fig_size=(12.80, 10.24)
    plt.title(name + '_IF')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    for i_dot in range(0, len(pca_result_DF)):
        this_S_str = pca_result_DF.index[i_dot].split('~')[0]
        this_S_int = int(this_S_str.split('S')[1])
        plot_X = pca_result_DF.iloc[i_dot, 0]
        plot_Y = pca_result_DF.iloc[i_dot, 1]
        # plt.plot(plot_X, plot_Y, color=dot_color_CHIR[i_dot], linestyle='-')
        plt.scatter(plot_X, plot_Y, color=dot_color_IF[this_S_int - 1])
        if text:
            text_X = plot_X
            text_Y = plot_Y
            plt.text(text_X, text_Y, this_S_str)
    fig.savefig(
        os.path.join(main_path, name + '_IF.png'))
    plt.close()

    return True


def draw_whole_picture(main_path, input_csv_path, output_png, show=False, D=2, shape='.', figsize=(128.0, 102.4),
                       x_min=None, x_max=None, y_min=None, y_max=None, text=False):
    # draw one whole pca picture using All_DATA_PCA.csv or All_FEATURES.csv
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_path:  r'C:\Users\Kitty\Desktop\CD13\All_DATA_PCA.csv'
    # output_png: 'do_draw.png'
    # D=2 : pca visualization dimension
    # shape='-' : the matplotlib plot shape '-' is line ; '.' is dot
    # x_min=None, x_max=None, y_min=None, y_max=None : the pca picture x y axis limit: plt.xlim

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    input_csv_path = os.path.join(main_path, input_csv_path)
    if not os.path.exists(input_csv_path):
        print('!ERROR! The input_csv_path does not existed!')
        return False

    pca_result_DF = pd.read_csv(input_csv_path, header=0, index_col=0)
    pca_result_DF = pca_result_DF.applymap(is_number)
    pca_result_DF = pca_result_DF.dropna(axis=0, how='any')
    pca_result = pca_result_DF.values

    well_name_list = []
    for i_str in pca_result_DF.index:
        well_name_list.append(int(i_str.split('~')[0].split('S')[1]))  # 'S1~2018-11-28~IPS_CD13~T1'
    well_name_S = pd.Series(well_name_list, name='S_name')
    well_count_S = well_name_S.groupby(well_name_S).count()  # always 96 wells
    # well_name_DF = pd.DataFrame(well_name_list,columns=['S_name'])
    # well_count_DF = well_name_DF.groupby(well_name_DF['S_name']).count()

    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    fig = plt.figure(figsize=figsize)
    i_index = 0
    for i in range(0, len(well_count_S)):  # for each well
        i_range = range(i_index, i_index + well_count_S.values[i])
        i_index = i_index + well_count_S.values[i]

        c = range(1, well_count_S.values[i] + 1)  # the latest the yellow color
        # c = pca_result[i_range, 2]
        color = np.random.rand(3)  # random line color
        if D == 2:
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            if shape == '-':
                plt.plot(pca_result[i_range, 0], pca_result[i_range, 1], color=color, linestyle='-',
                         label='S' + str(well_count_S.index[i]))
                plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c)
            elif shape == '.':
                plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c, label='S' + str(well_count_S.index[i]))
            else:
                plt.plot(pca_result[i_range, 0], pca_result[i_range, 1], color=color, marker=shape,
                         label='S' + str(well_count_S.index[i]))
            if text:
                for j in range(0, well_count_S.values[i]):
                    k = i_range[j]
                    plt.text(pca_result[k, 0], pca_result[k, 1], str(j + 1))
        elif D == 3:
            ax = plt.axes(projection='3d')
            ax.plot3D([pca_result[i_range, 0]], [pca_result[i_range, 1]], [pca_result[i_range, 2]], shape,
                      label='S' + str(well_count_S.index[i]))
        else:
            print('!ERROR! The D does not support!')
            return False

    # plt.title(title_str)
    # plt.legend(loc='upper right')
    fig.savefig(os.path.join(main_path, output_png))

    if show:
        plt.show()
    plt.close()

    return True


def draw_whole_picture_GroupBy__template(input_csv_path, output_png, input_exp_file, show=False,
                                         figsize=(12.80, 10.24),
                                         x_min=None, x_max=None, y_min=None, y_max=None, text=False):
    # the code template
    # draw a Conditionally filtered figure of some wells of one manifold file of one experiment
    # templateinput: input_csv_path, output_png, input_exp_file

    pca_result_DF = pd.read_csv(input_csv_path, header=0, index_col=0)
    pca_result_DF = pca_result_DF.applymap(is_number)
    pca_result_DF = pca_result_DF.dropna(axis=0, how='any')
    pca_result = pca_result_DF.values
    exp_DF = pd.read_csv(input_exp_file, header=0, index_col=0)

    well_name_list = []
    for i_str in pca_result_DF.index:
        well_name_list.append(int(i_str.split('~')[0].split('S')[1]))  # 'S1~2018-11-28~IPS_CD13~T1'
    well_name_S = pd.Series(well_name_list, name='S_name')
    well_count_S = well_name_S.groupby(well_name_S).count()  # always 96 wells
    well_count = well_count_S.shape[0]  # always 96 wells
    # well_name_DF = pd.DataFrame(well_name_list,columns=['S_name'])
    # well_count_DF = well_name_DF.groupby(well_name_DF['S_name']).count()

    IF_result_list = []  # from 0
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        IF_result_list.append(exp_DF.loc[i_S, 'IF_human'])

    CHIR_list = []
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        CHIR_list.append(exp_DF.loc[i_S, 'chir'])

    TIME_list = []
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        TIME_list.append(exp_DF.loc[i_S, 'chir_hour'])

    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    fig = plt.figure(figsize=figsize)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    i_index = 0
    for i in range(0, len(well_count_S)):  # for each well, i is well, from 0
        this_well_range = range(i_index, i_index + well_count_S.values[i])
        i_index = i_index + well_count_S.values[i]

        c = range(1, well_count_S.values[i] + 1)  # the latest the yellow color

        # if i == 0:  # this well
        #     # plt.plot(pca_result[this_well_range, 0], pca_result[this_well_range, 1], linestyle='-',
        #     #          label='S' + str(well_count_S.index[i]))
        #     plt.scatter(pca_result[this_well_range, 0], pca_result[this_well_range, 1], c=c,
        #                 label='S' + str(well_count_S.index[i]))
        #     if text:
        #         for j in range(0, well_count_S.values[i]):  # j is the time points range of this well
        #             k = this_well_range[j]
        #             plt.text(pca_result[k, 0], pca_result[k, 1], 'T' + str(j + 1))

        if IF_result_list[i] >= 0.5:  # this well
            plt.scatter(pca_result[this_well_range, 0], pca_result[this_well_range, 1], c=c,
                        label='S' + str(well_count_S.index[i]))
            if text:
                for j in range(0, well_count_S.values[i]):
                    k = this_well_range[j]
                    plt.text(pca_result[k, 0], pca_result[k, 1], str(j + 1))

    # plt.title(title_str)
    # plt.legend(loc='upper right')
    fig.savefig(output_png)

    if show:
        plt.show()
    plt.close()

    return True


def CD13_All_Success_wells_IFhuman_GE05(input_csv_path, output_png, input_exp_file, show=False,
                                        figsize=(12.80, 10.24),
                                        x_min=None, x_max=None, y_min=None, y_max=None, text=False):
    # for CD13 all Success wells

    pca_result_DF = pd.read_csv(input_csv_path, header=0, index_col=0)
    pca_result_DF = pca_result_DF.applymap(is_number)
    pca_result_DF = pca_result_DF.dropna(axis=0, how='any')
    pca_result = pca_result_DF.values
    exp_DF = pd.read_csv(input_exp_file, header=0, index_col=0)

    well_name_list = []
    for i_str in pca_result_DF.index:
        well_name_list.append(int(i_str.split('~')[0].split('S')[1]))  # 'S1~2018-11-28~IPS_CD13~T1'
    well_name_S = pd.Series(well_name_list, name='S_name')
    well_count_S = well_name_S.groupby(well_name_S).count()  # always 96 wells
    well_count = well_count_S.shape[0]  # always 96 wells

    IF_result_list = []  # from 0
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        IF_result_list.append(exp_DF.loc[i_S, 'IF_human'])

    # CHIR_list = []
    # for i in range(1, well_count + 1):
    #     i_S = 'S' + str(i)
    #     CHIR_list.append(exp_DF.loc[i_S, 'chir'])
    #
    # TIME_list = []
    # for i in range(1, well_count + 1):
    #     i_S = 'S' + str(i)
    #     TIME_list.append(exp_DF.loc[i_S, 'chir_hour'])

    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    fig = plt.figure(figsize=figsize)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    i_index = 0
    for i in range(0, len(well_count_S)):  # for each well, i is well, from 0
        this_well_range = range(i_index, i_index + well_count_S.values[i])
        i_index = i_index + well_count_S.values[i]

        c = range(1, well_count_S.values[i] + 1)  # the latest the yellow color

        if IF_result_list[i] >= 0.5:  # this well
            plt.scatter(pca_result[this_well_range, 0], pca_result[this_well_range, 1], c=c,
                        label='S' + str(well_count_S.index[i]))
            if text:
                for j in range(0, well_count_S.values[i]):
                    k = this_well_range[j]
                    plt.text(pca_result[k, 0], pca_result[k, 1], str(j + 1))

    fig.savefig(output_png)

    if show:
        plt.show()
    plt.close()

    return True


def CD13_All_Failure_wells_IFhuman_L01(input_csv_path, output_png, input_exp_file, show=False,
                                       figsize=(12.80, 10.24),
                                       x_min=None, x_max=None, y_min=None, y_max=None, text=False):
    # for CD13 all Success wells

    pca_result_DF = pd.read_csv(input_csv_path, header=0, index_col=0)
    pca_result_DF = pca_result_DF.applymap(is_number)
    pca_result_DF = pca_result_DF.dropna(axis=0, how='any')
    pca_result = pca_result_DF.values
    exp_DF = pd.read_csv(input_exp_file, header=0, index_col=0)

    well_name_list = []
    for i_str in pca_result_DF.index:
        well_name_list.append(int(i_str.split('~')[0].split('S')[1]))  # 'S1~2018-11-28~IPS_CD13~T1'
    well_name_S = pd.Series(well_name_list, name='S_name')
    well_count_S = well_name_S.groupby(well_name_S).count()  # always 96 wells
    well_count = well_count_S.shape[0]  # always 96 wells

    IF_result_list = []  # from 0
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        IF_result_list.append(exp_DF.loc[i_S, 'IF_human'])

    # CHIR_list = []
    # for i in range(1, well_count + 1):
    #     i_S = 'S' + str(i)
    #     CHIR_list.append(exp_DF.loc[i_S, 'chir'])
    #
    # TIME_list = []
    # for i in range(1, well_count + 1):
    #     i_S = 'S' + str(i)
    #     TIME_list.append(exp_DF.loc[i_S, 'chir_hour'])

    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    fig = plt.figure(figsize=figsize)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    i_index = 0
    for i in range(0, len(well_count_S)):  # for each well, i is well, from 0
        this_well_range = range(i_index, i_index + well_count_S.values[i])
        i_index = i_index + well_count_S.values[i]

        c = range(1, well_count_S.values[i] + 1)  # the latest the yellow color

        if IF_result_list[i] < 0.1:  # this well
            plt.scatter(pca_result[this_well_range, 0], pca_result[this_well_range, 1], c=c,
                        label='S' + str(well_count_S.index[i]))
            if text:
                for j in range(0, well_count_S.values[i]):
                    k = this_well_range[j]
                    plt.text(pca_result[k, 0], pca_result[k, 1], str(j + 1))

    fig.savefig(output_png)

    if show:
        plt.show()
    plt.close()

    return True


def do_draw_whole_stage_colored(main_path, input_csv_path, output_png, IF_file, draw=False, D=2,
                                figsize=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None, text=False):
    # 2: draw whole pca picture using '.' Separate into IPS I II III 'b' 'c' 'y' 'm'
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_path:  r'C:\Users\Kitty\Desktop\CD13\All_DATA_PCA.csv'
    # output_png: 'do_draw.png'
    # D=2 : pca visualization dimension
    # x_min=None, x_max=None, y_min=None, y_max=None : the pca picture x y axis limit: plt.xlim

    shape = '.'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_path):
        print('!ERROR! The input_csv_path does not existed!')
        return False

    IF_result = None
    if os.path.exists(os.path.join(main_path, IF_file)):
        IF_result = pd.read_csv(os.path.join(main_path, IF_file), header=0, index_col=0)

    pca_result_DF = pd.read_csv(input_csv_path, header=0, index_col=0)
    pca_result_DF = pca_result_DF.applymap(is_number)
    pca_result_DF = pca_result_DF.dropna(axis=0, how='any')
    pca_result = pca_result_DF.values

    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    fig = plt.figure(figsize=figsize)
    for i in range(0, len(pca_result_DF)):

        if pca_result_DF.index[i].split('~')[2].split('_')[0].split('-')[0] == 'IPS':
            color = 'b'
            label = 'IPS'
        elif pca_result_DF.index[i].split('~')[2].split('_')[0].split('-')[0] == 'I':
            color = 'c'  # 'springgreen'  # 'c'
            label = 'I'
        elif pca_result_DF.index[i].split('~')[2].split('_')[0].split('-')[0] == 'II':
            color = 'y'  # ''greenyellow'
            label = 'II'
        elif pca_result_DF.index[i].split('~')[2].split('_')[0].split('-')[0] == 'III':
            color = 'm'  # 'yellow'
            label = 'III'
        else:
            print('!ERROR! not valid state!')
            return False

        if D == 2:
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.plot(pca_result[i, 0], pca_result[i, 1], color=color, marker=shape, label=label)
            if text:
                S_int = int(pca_result_DF.index[i].split('~')[0].split('S')[1])
                plt.text(pca_result[i, 0], pca_result[i, 1], str(IF_result.loc['S' + str(S_int)].values[0]))
        elif D == 3:
            ax = plt.axes(projection='3d')
            ax.plot3D([pca_result[i, 0]], [pca_result[i, 1]], [pca_result[i, 2]], color=color, marker=shape,
                      label=label)
        else:
            print('!ERROR! The D does not support!')
            return False

    # plt.legend(loc='upper right')
    fig.savefig(os.path.join(main_path, output_png))

    if draw:
        plt.show()
    plt.close()

    return True


def do_draw_whole_result_colored_CD13_144h(main_path, input_csv_path, output_png, IF_file, draw=False, D=2,
                                           figsize=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None,
                                           text=False):
    # 3: draw day6: 0-0.2:'k' 0.2-0.4:'navy' 0.4-0.6:'gold' 0.6-0.8:'darkorange' 0.8-1:'r'
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_path:  r'C:\Users\Kitty\Desktop\CD13\All_DATA_PCA.csv'
    # output_png: 'do_draw.png'
    # D=2 : pca visualization dimension
    # x_min=None, x_max=None, y_min=None, y_max=None : the pca picture x y axis limit: plt.xlim

    shape = '.'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_path):
        print('!ERROR! The input_csv_path does not existed!')
        return False

    IF_result = None
    if os.path.exists(os.path.join(main_path, IF_file)):
        IF_result = pd.read_csv(os.path.join(main_path, IF_file), header=0, index_col=0)

    pca_result_DF = pd.read_csv(input_csv_path, header=0, index_col=0)
    pca_result_DF = pca_result_DF.applymap(is_number)
    pca_result_DF = pca_result_DF.dropna(axis=0, how='any')
    pca_result = pca_result_DF.values

    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    fig = plt.figure(figsize=figsize)
    for i in range(0, len(pca_result_DF)):

        if pca_result_DF.index[i].split('~')[2].split('_')[0] == 'II-3' and pca_result_DF.index[i].split('~')[
            3] == 'T44':
            S_int = int(pca_result_DF.index[i].split('~')[0].split('S')[1])
            if_intensity = IF_result.loc['S' + str(S_int)].values[0]
            if if_intensity <= 0.2:
                color = 'k'
            elif if_intensity <= 0.4:
                color = 'navy'
            elif if_intensity <= 0.6:
                color = 'gold'
            elif if_intensity <= 0.8:
                color = 'darkorange'
            else:
                color = 'r'
            if D == 2:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.plot(pca_result[i, 0], pca_result[i, 1], color=color, marker=shape, label=S_int)
                if text:
                    plt.text(pca_result[i, 0], pca_result[i, 1], str(if_intensity))
            elif D == 3:
                ax = plt.axes(projection='3d')
                ax.plot3D([pca_result[i, 0]], [pca_result[i, 1]], [pca_result[i, 2]], color=color, marker=shape,
                          label=S_int)
            else:
                print('!ERROR! The D does not support!')
                return False

        print('finish dot:', i)

    plt.legend(loc='upper right')
    fig.savefig(os.path.join(main_path, output_png))

    if draw:
        plt.show()
    plt.close()

    return True


def do_draw_dot(main_path, input_csv_path, output_fold, IF_file, figsize=(12.80, 10.24), x_min=None, x_max=None,
                y_min=None, y_max=None, text=False):
    # draw success\fail time Series dot flow pictures !only draw one time point dot !
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\PCA.csv'
    # output_fold = 'Figure_dot_flow'
    # IF_file =  r'C:\Users\Kitty\Desktop\CD13\IF_Result_human.csv'
    # D=2 : pca visualization dimension
    # shape='.' : only has one time point dot
    # x_min=None, x_max=None, y_min=None, y_max=None : the pca picture x y axis limit: plt.xlim

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_path):
        print('!ERROR! The input_csv_path does not existed!')
        return False
    if not os.path.exists(IF_file):
        print('!ERROR! The IF_file does not existed!')
        return False

    if os.path.exists(os.path.join(main_path, output_fold)):
        shutil.rmtree(os.path.join(main_path, output_fold))
    os.makedirs(os.path.join(main_path, output_fold))

    IF_result = pd.read_csv(os.path.join(main_path, IF_file), header=0, index_col=0)
    pca_result_DF = pd.read_csv(input_csv_path, header=0, index_col=0)
    pca_result_DF = pca_result_DF.applymap(is_number)
    pca_result_DF = pca_result_DF.dropna(axis=0, how='any')
    pca_result = pca_result_DF.values

    well_name_list = []
    for i_str in pca_result_DF.index:
        well_name_list.append(int(i_str.split('~')[0].split('S')[1]))
    well_name_S = pd.Series(well_name_list, name='S_name')
    well_count_S = well_name_S.groupby(well_name_S).count()  # the Series: number pic of each well
    offset_list = [0] * len(well_count_S)
    time_point = well_count_S.values[0]  # all shot time point
    time_point_benchmark_index = None

    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    for i_T in range(0, time_point):
        pca123 = ['pca' + str(col) for col in range(1, pca_result.shape[1] + 1)]
        pca123.append('Result')
        this_T_data = pd.DataFrame(columns=pca123)
        i_index = 0
        for i_S in range(0, len(well_count_S)):
            this_S_name = 'S' + str(well_count_S.index[i_S])  # well name exp:'S1'
            this_S_range = range(i_index, i_index + well_count_S.values[i_S])  # the iloc index range of pca_result_DF
            i_index += well_count_S.values[i_S]  # this start index auto update exp:0
            # this_S_range[i_T] : this pca index
            # offset_list[i_S] : this well offset
            if_intensity = IF_result.loc[this_S_name, 'IF_intensity']  # this well IF_result exp:0.9
            if time_point_benchmark_index is None:
                if this_S_name == 'S1':
                    time_point_benchmark_index = pca_result_DF.index[this_S_range]
                else:
                    print('!ERROR! The First well S1 does not existed!')
                    return False
            # this index exp:'S1~2018-11-28~IPS_CD13~T1'
            if i_T - offset_list[i_S] >= len(this_S_range):
                offset_list[i_S] += 1
                continue
            this_index_str = pca_result_DF.index[this_S_range[i_T - offset_list[i_S]]]

            bchmk_i_lst = time_point_benchmark_index[i_T].split('~')
            t_i_lst = this_index_str.split('~')
            if bchmk_i_lst[2] == t_i_lst[2] and bchmk_i_lst[3] == t_i_lst[3]:
                row_temp_ndarray = np.append(pca_result[this_S_range[i_T - offset_list[i_S]]], if_intensity)
                row_temp_DF = pd.DataFrame([row_temp_ndarray], columns=pca123, index=[this_index_str])
                this_T_data = this_T_data.append(row_temp_DF)
            else:
                offset_list[i_S] += 1
                continue
        # print(i_T, offset_list)
        fig = plt.figure(figsize=figsize)
        plt.title(time_point_benchmark_index[i_T].split('S1~')[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        for i_dot in range(len(this_T_data)):
            if this_T_data.iloc[i_dot, -1] <= 0.2:
                this_color = 'k'
            elif this_T_data.iloc[i_dot, -1] <= 0.4:
                this_color = 'navy'
            elif this_T_data.iloc[i_dot, -1] <= 0.6:
                this_color = 'gold'
            elif this_T_data.iloc[i_dot, -1] <= 0.8:
                this_color = 'darkorange'
            else:
                this_color = 'r'
            plt.plot(this_T_data.iloc[i_dot, 0], this_T_data.iloc[i_dot, 1], color=this_color, marker='.',
                     label=this_T_data.index[i_dot].split('~')[0])
            if text:
                plt.text(this_T_data.iloc[i_dot, 0], this_T_data.iloc[i_dot, 1], str(i_dot[-1]))
        # plt.legend(loc='upper right')
        fig.savefig(os.path.join(main_path, output_fold, time_point_benchmark_index[i_T].split('S1~')[1] + '.png'))
        plt.close()

    return True


def do_draw_dot_flow_3point_result_colored(main_path, input_csv_path, output_fold, IF_file, fig_size=(12.80, 10.24),
                                           x_min=None, x_max=None, y_min=None, y_max=None, text=False):
    # draw success\fail time Series dot flow pictures !draw 3 time point dot and '-'!
    # 3: using 3 of '-'
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\PCA.csv'
    # output_fold = 'Figure_dot_flow_2'
    # IF_file =  r'C:\Users\Kitty\Desktop\CD13\IF_Result_human.csv'
    # D=2 : pca visualization dimension
    # shape='-' : the matplotlib plot shape '-' is line ; '.' is dot
    # x_min=None, x_max=None, y_min=None, y_max=None : the pca picture x y axis limit: plt.xlim

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_path):
        print('!ERROR! The input_csv_path does not existed!')
        return False
    if not os.path.exists(IF_file):
        print('!ERROR! The IF_file does not existed!')
        return False

    if os.path.exists(os.path.join(main_path, output_fold)):
        shutil.rmtree(os.path.join(main_path, output_fold))
    os.makedirs(os.path.join(main_path, output_fold))

    IF_result = pd.read_csv(IF_file, header=0, index_col=0)
    pca_result_DF = pd.read_csv(input_csv_path, header=0, index_col=0)
    pca_result_DF = pca_result_DF.applymap(is_number)
    pca_result_DF = pca_result_DF.dropna(axis=0, how='any')
    pca_result = pca_result_DF.values

    well_name_list = []
    for i_str in pca_result_DF.index:
        well_name_list.append(int(i_str.split('~')[0].split('S')[1]))
    well_name_S = pd.Series(well_name_list, name='S_name')
    well_count_S = well_name_S.groupby(well_name_S).count()  # the Series: number pic of each well
    offset_list = [0] * len(well_count_S)
    time_point = well_count_S.values[0]  # all shot time point
    time_point_benchmark_index = None

    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    new_data_DF = None
    if_list = []

    for i_T in range(0, time_point):
        this_T_data = pd.DataFrame()
        i_index = 0
        for i_S in range(0, len(well_count_S)):
            this_S_name = 'S' + str(well_count_S.index[i_S])  # well name exp:'S1'
            this_S_range = range(i_index, i_index + well_count_S.values[i_S])  # the iloc index range of pca_result_DF
            i_index += well_count_S.values[i_S]  # this start index auto update exp:0
            # this_S_range[i_T] : this pca index
            # offset_list[i_S] : this well offset
            if_intensity = IF_result.loc[this_S_name, 'IF_intensity']  # this well IF_result exp:0.9
            if i_T == 0:
                if_list.append(if_intensity)
            if time_point_benchmark_index is None:
                if this_S_name == 'S1':
                    time_point_benchmark_index = pca_result_DF.index[this_S_range]
                else:
                    print('!ERROR! The First well S1 does not existed!')
                    return False
            # this index exp:'S1~2018-11-28~IPS_CD13~T1'
            if i_T - offset_list[i_S] >= len(this_S_range):
                offset_list[i_S] += 1
                continue
            this_index_str = pca_result_DF.index[this_S_range[i_T - offset_list[i_S]]]

            bchmk_i_lst = time_point_benchmark_index[i_T].split('~')
            t_i_lst = this_index_str.split('~')
            if bchmk_i_lst[2] == t_i_lst[2] and bchmk_i_lst[3] == t_i_lst[3]:
                # row_temp_ndarray = np.append(pca_result[this_S_range[i_T - offset_list[i_S]]], if_intensity)
                row_temp_ndarray = pca_result[this_S_range[i_T - offset_list[i_S]]]
                row_temp_DF = pd.DataFrame([row_temp_ndarray], index=[this_index_str])
                this_T_data = this_T_data.append(row_temp_DF)
            else:
                offset_list[i_S] += 1
                continue
        # print(this_T_data)
        index_tmp_list = []
        for it in this_T_data.index:
            # '~'.join(it.split('~')[1:-1])
            index_tmp_list.append(int(it.split('~')[0].split('S')[1]))
        this_T_data.index = index_tmp_list
        # print(this_T_data)
        if new_data_DF is None:
            new_data_DF = this_T_data
        else:
            new_data_DF = pd.concat([new_data_DF, this_T_data], axis=1)

    new_data_DF.to_csv(path_or_buf=os.path.join(main_path, 'Time_Series_PCA.csv'))

    # now draw part :::
    # new_data_DF = pd.read_csv(os.path.join(main_path, 'Time_Series_PCA.csv'), header=0, index_col=0)
    dot_color = []
    for i_intensity in if_list:
        if i_intensity <= 0.2:
            dot_color.append('k')
        elif i_intensity <= 0.4:
            dot_color.append('navy')
        elif i_intensity <= 0.6:
            dot_color.append('gold')
        elif i_intensity <= 0.8:
            dot_color.append('darkorange')
        else:
            dot_color.append('r')

    for i_T in range(0, time_point):  # i_T from 0 to 215
        fig = plt.figure(figsize=fig_size)  # fig_size=(12.80, 10.24)
        plt.title(time_point_benchmark_index[i_T].split('S1~')[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if i_T == 0:
            i_T_l = [0]
        elif i_T == 1:
            i_T_l = [0, 1]
        else:
            i_T_l = [i_T - 2, i_T - 1, i_T]
        # now begin plot '.' and '-'
        for i_dot in range(0, len(new_data_DF)):
            plot_X = new_data_DF.iloc[i_dot, [i * 3 for i in i_T_l]].values
            plot_Y = new_data_DF.iloc[i_dot, [i * 3 + 1 for i in i_T_l]].values
            plot_X = plot_X[~np.isnan(plot_X)]
            plot_Y = plot_Y[~np.isnan(plot_Y)]
            # print('plot_X:', plot_X, 'plot_Y:', plot_Y)
            # plt.plot(plot_X, plot_Y, color=dot_color[i_dot], marker='.', label='S'+str(new_data_DF.index[i_dot]))
            plt.plot(plot_X, plot_Y, color=dot_color[i_dot], linestyle='-', label='S' + str(new_data_DF.index[i_dot]))
            plt.scatter(plot_X, plot_Y, color=dot_color[i_dot])
            if text:
                text_X = plot_X[0]
                text_Y = plot_Y[0]
                plt.text(text_X, text_Y, 'S' + str(new_data_DF.index[i_dot]))
        # plt.legend(loc='upper right')
        fig.savefig(os.path.join(main_path, output_fold, time_point_benchmark_index[i_T].split('S1~')[1] + '.png'))
        plt.close()

    return True


def do_draw_dot_flow_3point_CHIR_colored_CD13(main_path, input_csv_vertical, input_csv_horizontal, output_fold,
                                              fig_size=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None,
                                              text=False):
    # different CHIR different color;different CHIR_time different shape
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\PCA.csv'
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\Time_Series_PCA.csv'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_vertical):
        print('!ERROR! The input_csv_vertical does not existed!')
        return False
    if not os.path.exists(input_csv_horizontal):
        print('!ERROR! The input_csv_horizontal does not existed!')
        return False
    if os.path.exists(os.path.join(main_path, output_fold)):
        shutil.rmtree(os.path.join(main_path, output_fold))
    os.makedirs(os.path.join(main_path, output_fold))

    pca_result_DF = pd.read_csv(input_csv_vertical, header=0, index_col=0)
    pca_result_DF = pca_result_DF.applymap(is_number)
    pca_result_DF = pca_result_DF.dropna(axis=0, how='any')
    pca_result = pca_result_DF.values
    if x_min is None:
        x_min = pca_result[:, 0].min()
    if x_max is None:
        x_max = pca_result[:, 0].max()
    if y_min is None:
        y_min = pca_result[:, 1].min()
    if y_max is None:
        y_max = pca_result[:, 1].max()

    new_data_DF = pd.read_csv(input_csv_horizontal, header=0, index_col=0)

    dot_color = []
    for i in range(0, 96):
        if i <= 23:  # chir 4
            dot_color.append('navy')
        elif i <= 47:  # chir 6
            dot_color.append('limegreen')  # 'limegreen'
        elif i <= 71:  # chir 8
            dot_color.append('gold')
        else:  # chir 10
            dot_color.append('r')
    # print(dot_color)
    # print(len(dot_color))
    time_shape = ['']

    well_name_list = []
    for i_str in pca_result_DF.index:
        well_name_list.append(int(i_str.split('~')[0].split('S')[1]))
    well_name_S = pd.Series(well_name_list, name='S_name')
    well_count_S = well_name_S.groupby(well_name_S).count()  # the Series: number pic of each well
    offset_list = [0] * len(well_count_S)
    time_point = well_count_S.values[0]  # all shot time point
    time_point_benchmark_index = pca_result_DF.index[range(0, time_point)]

    for i_T in range(0, time_point):  # i_T from 0 to 215
        fig = plt.figure(figsize=fig_size)  # fig_size=(12.80, 10.24)
        plt.title(time_point_benchmark_index[i_T].split('S1~')[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if i_T == 0:
            i_T_l = [0]
        elif i_T == 1:
            i_T_l = [0, 1]
        else:
            i_T_l = [i_T - 2, i_T - 1, i_T]
        # now begin plot '.' and '-'
        for i_dot in range(0, len(new_data_DF)):
            plot_X = new_data_DF.iloc[i_dot, [i * 3 for i in i_T_l]].values
            plot_Y = new_data_DF.iloc[i_dot, [i * 3 + 1 for i in i_T_l]].values
            plot_X = plot_X[~np.isnan(plot_X)]
            plot_Y = plot_Y[~np.isnan(plot_Y)]
            # print('plot_X:', plot_X, 'plot_Y:', plot_Y)
            # plt.plot(plot_X, plot_Y, color=dot_color[i_dot], marker='.', label='S'+str(new_data_DF.index[i_dot]))
            plt.plot(plot_X, plot_Y, color=dot_color[i_dot], linestyle='-', label='S' + str(new_data_DF.index[i_dot]))
            plt.scatter(plot_X, plot_Y, color=dot_color[i_dot])
            if text:
                text_X = plot_X[0]
                text_Y = plot_Y[0]
                plt.text(text_X, text_Y, 'S' + str(new_data_DF.index[i_dot]))
        # plt.legend(loc='upper right')
        fig.savefig(os.path.join(main_path, output_fold, time_point_benchmark_index[i_T].split('S1~')[1] + '.png'))
        plt.close()

    return True


def draw_dot_flow_3point(main_path, input_csv_vertical, input_csv_horizontal, exp_file, output_fold,
                         fig_size=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None, text=False):
    # different CHIR different color;different CHIR_time different shape
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\PCA.csv'
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\Time_Series_PCA.csv'
    # output_fold

    input_csv_vertical = os.path.join(main_path, input_csv_vertical)
    input_csv_horizontal = os.path.join(main_path, input_csv_horizontal)
    exp_file = os.path.join(main_path, exp_file)

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_vertical):
        print('!ERROR! The input_csv_vertical does not existed!')
        return False
    if not os.path.exists(input_csv_horizontal):
        print('!ERROR! The input_csv_horizontal does not existed!')
        return False
    if not os.path.exists(exp_file):
        print('!ERROR! The exp_file does not existed!')
        return False
    if os.path.exists(os.path.join(main_path, output_fold + '_CHIR')):
        shutil.rmtree(os.path.join(main_path, output_fold + '_CHIR'))
    os.makedirs(os.path.join(main_path, output_fold + '_CHIR'))
    if os.path.exists(os.path.join(main_path, output_fold + '_IF')):
        shutil.rmtree(os.path.join(main_path, output_fold + '_IF'))
    os.makedirs(os.path.join(main_path, output_fold + '_IF'))
    if os.path.exists(os.path.join(main_path, output_fold + '_TIME')):
        shutil.rmtree(os.path.join(main_path, output_fold + '_TIME'))
    os.makedirs(os.path.join(main_path, output_fold + '_TIME'))

    # if not isinstance(sp_time_point, list):
    #     sp_time_point = [sp_time_point]

    exp_DF = pd.read_csv(exp_file, header=0, index_col=0)
    vertical_DF = pd.read_csv(input_csv_vertical, header=0, index_col=0)
    horizontal_DF = pd.read_csv(input_csv_horizontal, header=0, index_col=0)

    well_count = horizontal_DF.shape[0]
    time_point = int(horizontal_DF.shape[1] / 3)
    time_point_benchmark_index = vertical_DF.index[range(0, time_point)]

    if x_min is None:
        x_min = vertical_DF.values[:, 0].min()
    if x_max is None:
        x_max = vertical_DF.values[:, 0].max()
    if y_min is None:
        y_min = vertical_DF.values[:, 1].min()
    if y_max is None:
        y_max = vertical_DF.values[:, 1].max()

    IF_result_list = []
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        IF_result_list.append(exp_DF.loc[i_S, 'IF_human'])
    dot_color_IF = []
    for i in IF_result_list:
        if i <= 0.2:
            dot_color_IF.append('k')
        elif i <= 0.4:
            dot_color_IF.append('navy')
        elif i <= 0.6:
            dot_color_IF.append('gold')
        elif i <= 0.8:
            dot_color_IF.append('darkorange')
        else:
            dot_color_IF.append('r')

    CHIR_list = []
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        CHIR_list.append(exp_DF.loc[i_S, 'chir'])
    dot_color_CHIR = []
    for i in CHIR_list:
        if i == 2:
            dot_color_CHIR.append('navy')
        elif i == 4:
            dot_color_CHIR.append('dodgerblue')
        elif i == 6:
            dot_color_CHIR.append('limegreen')
        elif i == 8:
            dot_color_CHIR.append('gold')
        elif i == 10:
            dot_color_CHIR.append('orange')
        elif i == 12:
            dot_color_CHIR.append('red')
        else:
            dot_color_CHIR.append('black')

    TIME_list = []
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        TIME_list.append(exp_DF.loc[i_S, 'chir_hour'])
    dot_color_TIME = []
    for i in TIME_list:
        if i == 18:
            dot_color_TIME.append('navy')
        elif i == 24:
            dot_color_TIME.append('dodgerblue')
        elif i == 30:
            dot_color_TIME.append('limegreen')
        elif i == 36:
            dot_color_TIME.append('gold')
        elif i == 42:
            dot_color_TIME.append('orange')
        elif i == 48:
            dot_color_TIME.append('red')
        elif i == 54:
            dot_color_TIME.append('purple')
        else:
            dot_color_TIME.append('black')

    for i_T in range(0, time_point):  # i_T from 0 to 215
        fig = plt.figure(figsize=fig_size)  # fig_size=(12.80, 10.24)
        plt.title(time_point_benchmark_index[i_T].split('S1~')[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if i_T == 0:
            i_T_l = [0]
        elif i_T == 1:
            i_T_l = [0, 1]
        else:
            i_T_l = [i_T - 2, i_T - 1, i_T]
        # now begin plot '.' and '-'
        for i_dot in range(0, len(horizontal_DF)):
            plot_X = horizontal_DF.iloc[i_dot, [i * 3 for i in i_T_l]].values
            plot_Y = horizontal_DF.iloc[i_dot, [i * 3 + 1 for i in i_T_l]].values
            plot_X = plot_X[~np.isnan(plot_X)]
            plot_Y = plot_Y[~np.isnan(plot_Y)]
            # print('plot_X:', plot_X, 'plot_Y:', plot_Y)
            # plt.plot(plot_X, plot_Y, color=dot_color[i_dot], marker='.', label='S'+str(new_data_DF.index[i_dot]))
            plt.plot(plot_X, plot_Y, color=dot_color_CHIR[i_dot], linestyle='-',
                     label='S' + str(horizontal_DF.index[i_dot]))
            plt.scatter(plot_X, plot_Y, color=dot_color_CHIR[i_dot])
            if text:
                text_X = plot_X[0]
                text_Y = plot_Y[0]
                plt.text(text_X, text_Y, 'S' + str(horizontal_DF.index[i_dot]))
        # plt.legend(loc='upper right')
        fig.savefig(
            os.path.join(main_path, output_fold + '_CHIR', time_point_benchmark_index[i_T].split('S1~')[1] + '.png'))
        plt.close()

    for i_T in range(0, time_point):  # i_T from 0 to 215
        fig = plt.figure(figsize=fig_size)  # fig_size=(12.80, 10.24)
        plt.title(time_point_benchmark_index[i_T].split('S1~')[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if i_T == 0:
            i_T_l = [0]
        elif i_T == 1:
            i_T_l = [0, 1]
        else:
            i_T_l = [i_T - 2, i_T - 1, i_T]
        # now begin plot '.' and '-'
        for i_dot in range(0, len(horizontal_DF)):
            plot_X = horizontal_DF.iloc[i_dot, [i * 3 for i in i_T_l]].values
            plot_Y = horizontal_DF.iloc[i_dot, [i * 3 + 1 for i in i_T_l]].values
            plot_X = plot_X[~np.isnan(plot_X)]
            plot_Y = plot_Y[~np.isnan(plot_Y)]
            # print('plot_X:', plot_X, 'plot_Y:', plot_Y)
            # plt.plot(plot_X, plot_Y, color=dot_color[i_dot], marker='.', label='S'+str(new_data_DF.index[i_dot]))
            plt.plot(plot_X, plot_Y, color=dot_color_IF[i_dot], linestyle='-',
                     label='S' + str(horizontal_DF.index[i_dot]))
            plt.scatter(plot_X, plot_Y, color=dot_color_IF[i_dot])
            if text:
                text_X = plot_X[0]
                text_Y = plot_Y[0]
                plt.text(text_X, text_Y, 'S' + str(horizontal_DF.index[i_dot]))
        # plt.legend(loc='upper right')
        fig.savefig(
            os.path.join(main_path, output_fold + '_IF', time_point_benchmark_index[i_T].split('S1~')[1] + '.png'))
        plt.close()

    for i_T in range(0, time_point):  # i_T from 0 to 215
        fig = plt.figure(figsize=fig_size)  # fig_size=(12.80, 10.24)
        plt.title(time_point_benchmark_index[i_T].split('S1~')[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if i_T == 0:
            i_T_l = [0]
        elif i_T == 1:
            i_T_l = [0, 1]
        else:
            i_T_l = [i_T - 2, i_T - 1, i_T]
        # now begin plot '.' and '-'
        for i_dot in range(0, len(horizontal_DF)):
            plot_X = horizontal_DF.iloc[i_dot, [i * 3 for i in i_T_l]].values
            plot_Y = horizontal_DF.iloc[i_dot, [i * 3 + 1 for i in i_T_l]].values
            plot_X = plot_X[~np.isnan(plot_X)]
            plot_Y = plot_Y[~np.isnan(plot_Y)]
            # print('plot_X:', plot_X, 'plot_Y:', plot_Y)
            # plt.plot(plot_X, plot_Y, color=dot_color[i_dot], marker='.', label='S'+str(new_data_DF.index[i_dot]))
            plt.plot(plot_X, plot_Y, color=dot_color_TIME[i_dot], linestyle='-',
                     label='S' + str(horizontal_DF.index[i_dot]))
            plt.scatter(plot_X, plot_Y, color=dot_color_TIME[i_dot])
            if text:
                text_X = plot_X[0]
                text_Y = plot_Y[0]
                plt.text(text_X, text_Y, 'S' + str(horizontal_DF.index[i_dot]))
        # plt.legend(loc='upper right')
        fig.savefig(
            os.path.join(main_path, output_fold + '_TIME', time_point_benchmark_index[i_T].split('S1~')[1] + '.png'))
        plt.close()

    return True


def new_draw_dot_flow_3point(main_path, input_csv_vertical, input_csv_horizontal, exp_file, output_fold,
                             fig_size=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None, text=False):
    # different CHIR different color;different CHIR_time different size
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\PCA.csv'
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\Time_Series_PCA.csv'
    # output_fold

    input_csv_vertical = os.path.join(main_path, input_csv_vertical)
    input_csv_horizontal = os.path.join(main_path, input_csv_horizontal)
    exp_file = os.path.join(main_path, exp_file)

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_vertical):
        print('!ERROR! The input_csv_vertical does not existed!')
        return False
    if not os.path.exists(input_csv_horizontal):
        print('!ERROR! The input_csv_horizontal does not existed!')
        return False
    if not os.path.exists(exp_file):
        print('!ERROR! The exp_file does not existed!')
        return False
    if os.path.exists(os.path.join(main_path, output_fold)):
        shutil.rmtree(os.path.join(main_path, output_fold))
    os.makedirs(os.path.join(main_path, output_fold))

    # if not isinstance(sp_time_point, list):
    #     sp_time_point = [sp_time_point]

    exp_DF = pd.read_csv(exp_file, header=0, index_col=0)
    vertical_DF = pd.read_csv(input_csv_vertical, header=0, index_col=0)
    horizontal_DF = pd.read_csv(input_csv_horizontal, header=0, index_col=0)

    well_count = horizontal_DF.shape[0]
    time_point = int(horizontal_DF.shape[1] / 3)
    time_point_benchmark_index = vertical_DF.index[range(0, time_point)]

    if x_min is None:
        x_min = vertical_DF.values[:, 0].min()
    if x_max is None:
        x_max = vertical_DF.values[:, 0].max()
    if y_min is None:
        y_min = vertical_DF.values[:, 1].min()
    if y_max is None:
        y_max = vertical_DF.values[:, 1].max()

    IF_result_list = []
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        IF_result_list.append(exp_DF.loc[i_S, 'IF_human'])
    dot_color_IF = []
    for i in IF_result_list:
        if i >= 0 and i <= 0.1:
            dot_color_IF.append('snow')
        elif i > 0.1 and i <= 0.2:
            dot_color_IF.append('snow')
        elif i > 0.2 and i <= 0.3:
            dot_color_IF.append('snow')
        elif i > 0.3 and i <= 0.4:
            dot_color_IF.append('snow')
        elif i > 0.4 and i <= 0.5:
            dot_color_IF.append('snow')
        elif i > 0.5 and i <= 0.6:
            dot_color_IF.append('mistyrose')
        elif i > 0.6 and i <= 0.7:
            dot_color_IF.append('pink')
        elif i > 0.7 and i <= 0.8:
            dot_color_IF.append('orangered')
        elif i > 0.8 and i <= 0.9:
            dot_color_IF.append('red')
        elif i > 0.9 and i <= 1:
            dot_color_IF.append('crimson')
        else:
            print('!WARNING! The Influence Intensity must between 0~1!')
            dot_color_IF.append('lightslategray')

    CHIR_list = []
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        CHIR_list.append(exp_DF.loc[i_S, 'chir'])
    dot_color_CHIR = []
    for i in CHIR_list:
        if i == 0:
            dot_color_CHIR.append('aliceblue')
        elif i > 0 and i <= 2:
            dot_color_CHIR.append('lightblue')
        elif i > 2 and i <= 4:
            dot_color_CHIR.append('deepskyblue')
        elif i > 4 and i <= 6:
            dot_color_CHIR.append('dodgerblue')
        elif i > 6 and i <= 8:
            dot_color_CHIR.append('cornflowerblue')
        elif i > 8 and i <= 10:
            dot_color_CHIR.append('royalblue')
        elif i > 10 and i <= 12:
            dot_color_CHIR.append('blue')
        elif i > 12 and i <= 14:
            dot_color_CHIR.append('mediumblue')
        elif i > 14:
            dot_color_CHIR.append('navy')
        else:
            print('!WARNING! The CHIR does not existed!')
            dot_color_CHIR.append('lightslategray')

    TIME_list = []
    for i in range(1, well_count + 1):
        i_S = 'S' + str(i)
        TIME_list.append(exp_DF.loc[i_S, 'chir_hour'])
    dot_color_TIME = []
    for i in TIME_list:
        if i == 18:
            dot_color_TIME.append('navy')
        elif i == 24:
            dot_color_TIME.append('dodgerblue')
        elif i == 30:
            dot_color_TIME.append('limegreen')
        elif i == 36:
            dot_color_TIME.append('yellow')
        elif i == 42:
            dot_color_TIME.append('orange')
        elif i == 48:
            dot_color_TIME.append('red')
        elif i == 54:
            dot_color_TIME.append('purple')
        else:
            dot_color_TIME.append('black')

    for i_T in range(0, time_point):  # i_T from 0 to 215
        fig = plt.figure(figsize=fig_size)  # fig_size=(12.80, 10.24)
        plt.title(time_point_benchmark_index[i_T].split('S1~')[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if i_T == 0:
            i_T_l = [0]
        elif i_T == 1:
            i_T_l = [0, 1]
        else:
            i_T_l = [i_T - 2, i_T - 1, i_T]
        # now begin plot '.' and '-'
        for i_dot in range(0, well_count):
            plot_X = horizontal_DF.iloc[i_dot, [i * 3 for i in i_T_l]].values
            plot_Y = horizontal_DF.iloc[i_dot, [i * 3 + 1 for i in i_T_l]].values
            plot_X = plot_X[~np.isnan(plot_X)]
            plot_Y = plot_Y[~np.isnan(plot_Y)]
            # print('plot_X:', plot_X, 'plot_Y:', plot_Y)
            # plt.plot(plot_X, plot_Y, color=dot_color[i_dot], marker='.', label='S'+str(new_data_DF.index[i_dot]))
            plt.plot(plot_X, plot_Y, color=dot_color_IF[i_dot], linestyle='-',
                     label='S' + str(horizontal_DF.index[i_dot]))
            plt.scatter(plot_X, plot_Y, color=dot_color_IF[i_dot], s=TIME_list[i_dot] * 3,
                        edgecolors=dot_color_CHIR[i_dot], linewidths=3,
                        alpha=0.9)
            if text:
                text_X = plot_X[0]
                text_Y = plot_Y[0]
                plt.text(text_X, text_Y, 'S' + str(horizontal_DF.index[i_dot]))
        # plt.legend(loc='upper right')
        fig.savefig(
            os.path.join(main_path, output_fold, time_point_benchmark_index[i_T].split('S1~')[1] + '.png'))
        plt.close()

    return True


def do_draw_dot_flow_exp_sep_3point_CD13(main_path, input_csv_vertical, input_csv_horizontal, IF_file, output_fold,
                                         exp_sep, fig_size=(12.80, 10.24), x_min=None, x_max=None, y_min=None,
                                         y_max=None, text=False):
    # different CHIR different color;different CHIR_time different shape
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\PCA.csv'
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\Time_Series_PCA.csv'
    # exp_sep = [0,11,24,42,74,107,125,170]
    # output_fold

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_vertical):
        print('!ERROR! The input_csv_vertical does not existed!')
        return False
    if not os.path.exists(input_csv_horizontal):
        print('!ERROR! The input_csv_horizontal does not existed!')
        return False
    if not os.path.exists(IF_file):
        print('!ERROR! The IF_file does not existed!')
        return False
    if os.path.exists(os.path.join(main_path, output_fold + '_CHIR')):
        shutil.rmtree(os.path.join(main_path, output_fold + '_CHIR'))
    os.makedirs(os.path.join(main_path, output_fold + '_CHIR'))
    if os.path.exists(os.path.join(main_path, output_fold + '_IF')):
        shutil.rmtree(os.path.join(main_path, output_fold + '_IF'))
    os.makedirs(os.path.join(main_path, output_fold + '_IF'))

    # if not isinstance(sp_time_point, list):
    #     sp_time_point = [sp_time_point]

    IF_result_DF = pd.read_csv(os.path.join(main_path, IF_file), header=0, index_col=0)
    pca_result_DF = pd.read_csv(input_csv_vertical, header=0, index_col=0)
    new_data_DF = pd.read_csv(input_csv_horizontal, header=0, index_col=0)

    if x_min is None:
        x_min = pca_result_DF.values[:, 0].min()
    if x_max is None:
        x_max = pca_result_DF.values[:, 0].max()
    if y_min is None:
        y_min = pca_result_DF.values[:, 1].min()
    if y_max is None:
        y_max = pca_result_DF.values[:, 1].max()

    if not isinstance(exp_sep, list):
        sp_tp = [exp_sep]

    IF_result_list = []
    for i in new_data_DF.index:
        i = 'S' + str(i)
        IF_result_list.append(IF_result_DF.loc[i, 'IF_intensity'])

    # CD13 CHIR distribution
    dot_color_CHIR = []
    for i in range(0, len(new_data_DF)):
        if i <= 23:  # chir 4
            dot_color_CHIR.append('navy')
        elif i <= 47:  # chir 6
            dot_color_CHIR.append('limegreen')
        elif i <= 71:  # chir 8
            dot_color_CHIR.append('gold')
        else:  # chir 10
            dot_color_CHIR.append('r')
    # CD13 CHIR distribution

    dot_color_IF = []
    for i in IF_result_list:
        if i <= 0.2:
            dot_color_IF.append('k')
        elif i <= 0.4:
            dot_color_IF.append('navy')
        elif i <= 0.6:
            dot_color_IF.append('gold')
        elif i <= 0.8:
            dot_color_IF.append('darkorange')
        else:
            dot_color_IF.append('r')

    # well_name_list = []
    # for i_str in pca_result_DF.index:
    #     well_name_list.append(int(i_str.split('~')[0].split('S')[1]))
    # well_name_S = pd.Series(well_name_list, name='S_name')
    # well_count_S = well_name_S.groupby(well_name_S).count()  # the Series: number pic of each well
    # time_point = well_count_S.values[0]  # all shot time point
    time_point = int(new_data_DF.shape[1] / 3)
    time_point_benchmark_index = pca_result_DF.index[range(0, time_point)]

    for i_T in range(0, time_point):  # i_T from 0 to 215
        fig = plt.figure(figsize=fig_size)  # fig_size=(12.80, 10.24)
        plt.title(time_point_benchmark_index[i_T].split('S1~')[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if i_T in exp_sep:
            i_T_l = [i_T]
        elif i_T - 1 in exp_sep:
            i_T_l = [i_T - 1, i_T]
        else:
            i_T_l = [i_T - 2, i_T - 1, i_T]
        # now begin plot '.' and '-'
        for i_dot in range(0, len(new_data_DF)):
            plot_X = new_data_DF.iloc[i_dot, [i * 3 for i in i_T_l]].values
            plot_Y = new_data_DF.iloc[i_dot, [i * 3 + 1 for i in i_T_l]].values
            plot_X = plot_X[~np.isnan(plot_X)]
            plot_Y = plot_Y[~np.isnan(plot_Y)]
            # print('plot_X:', plot_X, 'plot_Y:', plot_Y)
            # plt.plot(plot_X, plot_Y, color=dot_color[i_dot], marker='.', label='S'+str(new_data_DF.index[i_dot]))
            plt.plot(plot_X, plot_Y, color=dot_color_CHIR[i_dot], linestyle='-',
                     label='S' + str(new_data_DF.index[i_dot]))
            plt.scatter(plot_X, plot_Y, color=dot_color_CHIR[i_dot])
            if text:
                text_X = plot_X[0]
                text_Y = plot_Y[0]
                plt.text(text_X, text_Y, 'S' + str(new_data_DF.index[i_dot]))
        # plt.legend(loc='upper right')
        fig.savefig(
            os.path.join(main_path, output_fold + '_CHIR', time_point_benchmark_index[i_T].split('S1~')[1] + '.png'))
        plt.close()

    for i_T in range(0, time_point):  # i_T from 0 to 215
        fig = plt.figure(figsize=fig_size)  # fig_size=(12.80, 10.24)
        plt.title(time_point_benchmark_index[i_T].split('S1~')[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if i_T in exp_sep:
            i_T_l = [i_T]
        elif i_T - 1 in exp_sep:
            i_T_l = [i_T - 1, i_T]
        else:
            i_T_l = [i_T - 2, i_T - 1, i_T]
        # now begin plot '.' and '-'
        for i_dot in range(0, len(new_data_DF)):
            plot_X = new_data_DF.iloc[i_dot, [i * 3 for i in i_T_l]].values
            plot_Y = new_data_DF.iloc[i_dot, [i * 3 + 1 for i in i_T_l]].values
            plot_X = plot_X[~np.isnan(plot_X)]
            plot_Y = plot_Y[~np.isnan(plot_Y)]
            # print('plot_X:', plot_X, 'plot_Y:', plot_Y)
            # plt.plot(plot_X, plot_Y, color=dot_color[i_dot], marker='.', label='S'+str(new_data_DF.index[i_dot]))
            plt.plot(plot_X, plot_Y, color=dot_color_IF[i_dot], linestyle='-',
                     label='S' + str(new_data_DF.index[i_dot]))
            plt.scatter(plot_X, plot_Y, color=dot_color_IF[i_dot])
            if text:
                text_X = plot_X[0]
                text_Y = plot_Y[0]
                plt.text(text_X, text_Y, 'S' + str(new_data_DF.index[i_dot]))
        # plt.legend(loc='upper right')
        fig.savefig(
            os.path.join(main_path, output_fold + '_IF', time_point_benchmark_index[i_T].split('S1~')[1] + '.png'))
        plt.close()

    return True


def draw_mainfold(main_path, mainfold_path, exp_file):
    # go over mainfold_path mainfold files, and draw
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # mainfold_path = r'C:\Users\Kitty\Desktop\CD13\MainFold'
    # exp_file = r'C:\Users\Kitty\Desktop\CD13\Experiment_Plan.csv'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    mainfold_path = os.path.join(main_path, mainfold_path)
    if not os.path.exists(mainfold_path):
        print('!ERROR! The mainfold_path does not existed!')
        return False
    exp_file = os.path.join(main_path, exp_file)
    if not os.path.exists(exp_file):
        print('!ERROR! The exp_file does not existed!')
        return False

    analysis_CSVs = os.listdir(mainfold_path)
    for ana_csv in analysis_CSVs:
        if ana_csv[-4:] == '.csv' and ana_csv.find('_horizontal') == -1:
            this_v_file = os.path.join(mainfold_path, ana_csv)
            this_name = ana_csv[:-4]

            this_h_file = os.path.join(mainfold_path, this_name + '_horizontal' + '.csv')
            if not os.path.exists(this_h_file):
                pca_vertical_to_horizontal(mainfold_path, this_v_file, this_h_file)

            this_png_file = os.path.join(mainfold_path, this_name + '.png')
            if not os.path.exists(this_png_file):
                draw_whole_picture(mainfold_path, this_v_file, this_png_file, show=False, D=2, shape='.',
                                   figsize=(12.80 * 2, 10.24 * 2), x_min=None, x_max=None, y_min=None, y_max=None,
                                   text=False)

            if not os.path.exists(os.path.join(mainfold_path, this_name)):
                new_draw_dot_flow_3point(mainfold_path, this_v_file, this_h_file, exp_file, this_name,
                                         fig_size=(12.80 * 2, 10.24 * 2), x_min=None, x_max=None, y_min=None,
                                         y_max=None, text=False)

    return True


def draw_mainfold_each_whole_inOneFolder(main_path, mainfold_path):
    # go over mainfold files, and draw one whole manifolder picture
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # mainfold_path = r'C:\Users\Kitty\Desktop\CD13\MainFold'
    # exp_file = r'C:\Users\Kitty\Desktop\CD13\Experiment_Plan.csv'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    mainfold_path = os.path.join(main_path, mainfold_path)
    if not os.path.exists(mainfold_path):
        print('!ERROR! The mainfold_path does not existed!')
        return False

    analysis_CSVs = os.listdir(mainfold_path)
    for ana_csv in analysis_CSVs:
        if ana_csv[-4:] == '.csv' and ana_csv.find('_horizontal') == -1:
            this_v_file = os.path.join(mainfold_path, ana_csv)
            this_name = ana_csv[:-4]

            this_png_file = os.path.join(mainfold_path, this_name + '.png')
            if not os.path.exists(this_png_file):
                draw_whole_picture(mainfold_path, this_v_file, this_png_file, show=False, D=2, shape='.',
                                   figsize=(12.80 * 2, 10.24 * 2), x_min=None, x_max=None, y_min=None, y_max=None,
                                   text=False)

    return True


def draw_mainfold_elastic_inOneFolder_bat(main_path, mainfold_path, exp_file, function_list):
    # for each manifold csv file in mainfold_path, do drawing function in function_list
    # finally, we get (len(os.listdir(mainfold_path))*len(function_list)) figures!

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    mainfold_path = os.path.join(main_path, mainfold_path)
    if not os.path.exists(mainfold_path):
        print('!ERROR! The mainfold_path does not existed!')
        return False
    exp_file = os.path.join(main_path, exp_file)
    if not os.path.exists(exp_file):
        print('!ERROR! The exp_file does not existed!')
        return False
    if type(function_list) is list:
        if len(function_list) == 0:
            print('!ERROR! The function_list must have one function!')
            return False
        else:
            for each_function in function_list:
                if callable(each_function):
                    pass
                else:
                    print('!ERROR! The function_list must be function!')
                    return False
    elif callable(function_list):
        function_list = [function_list]
    else:
        print('!ERROR! The function_list must be function or function list!')
        return False

    manifold_CSVs = os.listdir(mainfold_path)
    for each_manifold_csv in manifold_CSVs:
        if each_manifold_csv[-4:] == '.csv' and each_manifold_csv.find('_horizontal') == -1:
            this_manifold_file = os.path.join(mainfold_path, each_manifold_csv)
            this_manifold_name = each_manifold_csv[:-4]

            for each_function in function_list:
                this_function_name = each_function.__name__
                this_png_file = os.path.join(mainfold_path, this_function_name + '_' + this_manifold_name + '.png')

                if not os.path.exists(this_png_file):
                    each_function(this_manifold_file, this_png_file, exp_file)

    return True


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Lib_Visualization.py !')

    main_path = r'C:\Users\Kitty\Desktop\CD13_Test_20210728'
    mainfold_path = r'MainFold_ALL'
    exp_file = r'Experiment_Plan.csv'
    function_list = [CD13_All_Success_wells_IFhuman_GE05, CD13_All_Failure_wells_IFhuman_L01]
    draw_mainfold_elastic_inOneFolder_bat(main_path, mainfold_path, exp_file, function_list)

    # main_path = r'C:\Users\Kitty\Desktop\CD13_Test_20210728'
    # mainfold_path = r'MainFold_ALL'
    # exp_file = r'Experiment_Plan.csv'
    # draw_mainfold_each_whole_inOneFolder(main_path, mainfold_path, exp_file)

    # main_path = r'C:\Users\Kitty\Documents\Desktop\CD30\PROCESSING'
    # exp_file = r'Experiment_Plan_B.csv'
    # mainfold_path = r'Classed_all_C8-60H_Features'
    # draw_mainfold(main_path, mainfold_path, exp_file)
    # mainfold_path = r'Classed_all_C8-60H_Enhanced_Features'
    # draw_mainfold(main_path, mainfold_path, exp_file)
    # mainfold_path = r'Classed_all_C8-60H_MyPGC_Features'
    # draw_mainfold(main_path, mainfold_path, exp_file)
    # draw_whole_picture(main_path, input_csv_path, output_png, show=False, D=2, shape='.', figsize=(128.0, 102.4),
    #                    x_min=None, x_max=None, y_min=None, y_max=None, text=False)

    # main_path = r'J:\PROCESSING\CD13'
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # features_path = r'C:\Users\Kitty\Desktop\CD13\Analysis'
    # IF_file = r'C:\Users\Kitty\Desktop\CD13\IF_Result_human.csv'
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\MainFold\Isomap.csv'
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\Time_Series_PCA.csv'
    # output_fold = 'PCA_dot_flow_3'

    # print('MainFold_ALL')
    # draw_mainfold(main_path, 'MainFold_ALL', 'Experiment_Plan.csv')
    # print('MainFold_ORB')
    # draw_mainfold(main_path, 'MainFold_ORB', 'Experiment_Plan.csv')
    # print('MainFold_SIFT')
    # draw_mainfold(main_path, 'MainFold_SIFT', 'Experiment_Plan.csv')
    # print('MainFold_SUR')
    # draw_mainfold(main_path, 'MainFold_SUR', 'Experiment_Plan.csv')

    # do_draw_whole_picture(main_path, input_csv_vertical, r'C:\Users\Kitty\Desktop\CD13\MainFold\Isomap.png', draw=False,
    #                       D=2, shape='.', figsize=(128.0, 102.4), x_min=None, x_max=None, y_min=None, y_max=None,
    #                       text=False)
    # exp_sep = [0, 11, 24, 42, 74, 107, 125, 170]
    # exp_sep = [0, 11, 24, 42, 60, 65, 70, 74, 78, 107, 117, 125, 170]
    #
    # do_draw_dot_flow_exp_sep_3point_CD13(main_path, input_csv_vertical, input_csv_horizontal, IF_file, output_fold,
    #                                      exp_sep, fig_size=(12.80, 10.24), x_min=-75, x_max=75, y_min=None,
    #                                      y_max=75, text=False)

    # do_draw_whole_picture(main_path, input_csv_path, 'PCA_time_colored_small.png', draw=False, D=2, shape='.',
    #                       figsize=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None, text=False)
    #
    # do_draw_whole_stage_colored(main_path, input_csv_path, 'PCA_stage_colored_small_3D.png', IF_file, draw=False, D=2,
    #                             figsize=(12.80, 10.24), x_min=-75, x_max=75, y_min=None, y_max=75, text=False)
    # # Day3 begin (72h)
    # sp_tp = ['2018-12-03~I-2_CD13~T18']  # Day3 begin (72h)
    # merge_specific_time_point_features(main_path, features_path, sp_tp, 'features_72H.csv')
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\features_72H.csv'
    # do_pca(main_path, input_csv_path, 'PCA_72H.csv', MAX_mle=3)
    # input_csv = r'C:\Users\Kitty\Desktop\CD13\PCA_72H.csv'
    # do_draw_one_time_CD13(main_path, input_csv, IF_file, figsize=(12.80, 10.24))
    # # Day3 begin (72h)
    #
    # # Day5 end (day6 begin)
    # sp_tp = ['2018-12-06~II-2_CD13~T18']  # Day5 end (day6 begin)(144h)
    # merge_specific_time_point_features(main_path, features_path, sp_tp, 'features_144H.csv')
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\features_144H.csv'
    # do_pca(main_path, input_csv_path, 'PCA_144H.csv', MAX_mle=3)
    # input_csv = r'C:\Users\Kitty\Desktop\CD13\PCA_144H.csv'
    # do_draw_one_time_CD13(main_path, input_csv, IF_file, figsize=(12.80, 10.24))
    #
    # # Day5 end (day6 begin)
    #
    # # Day11 end (result)
    # sp_tp = ['2018-12-10~III-1_CD13~T45']  # Day11 end (result)
    # merge_specific_time_point_features(main_path, features_path, sp_tp, 'features_all_end.csv')
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\features_all_end.csv'
    # do_pca(main_path, input_csv_path, 'PCA_all_end.csv', MAX_mle=3)
    # input_csv = r'C:\Users\Kitty\Desktop\CD13\PCA_all_end.csv'
    # do_draw_one_time_CD13(main_path, input_csv, IF_file, figsize=(12.80, 10.24))
    # # Day11 end (result)
    #
    # # CHIR
    # sp_tp = []  # CHIR
    # for i in range(1, 16):
    #     sp_tp.append('2018-12-01~I-1_CD13~T' + str(i))
    # merge_specific_time_point_features(main_path, features_path, sp_tp, 'features_add_CHIR.csv')
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\features_add_CHIR.csv'
    # do_pca(main_path, input_csv_path, 'PCA_vertical_add_CHIR.csv', MAX_mle=3)
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\PCA_vertical_add_CHIR.csv'
    # pca_vertical_to_horizontal(main_path, input_csv_vertical, 'PCA_horizontal_add_CHIR.csv')
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\PCA_horizontal_add_CHIR.csv'
    # do_draw_dot_flow_3point_CD13(main_path, input_csv_vertical, input_csv_horizontal, IF_file, 'PCA_add_CHIR',
    #                              fig_size=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None, text=False)
    # # CHIR
    #
    # # CHIR rest
    # sp_tp = []  # CHIR rest
    # for i in range(5, 19):
    #     sp_tp.append('2018-12-03~I-2_CD13~T' + str(i))
    # merge_specific_time_point_features(main_path, features_path, sp_tp, 'features_CHIR_rest.csv')
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\features_CHIR_rest.csv'
    # do_pca(main_path, input_csv_path, 'PCA_vertical_CHIR_rest.csv', MAX_mle=3)
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\PCA_vertical_CHIR_rest.csv'
    # pca_vertical_to_horizontal(main_path, input_csv_vertical, 'PCA_horizontal_CHIR_rest.csv')
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\PCA_horizontal_CHIR_rest.csv'
    # do_draw_dot_flow_3point_CD13(main_path, input_csv_vertical, input_csv_horizontal, IF_file, 'PCA_CHIR_rest',
    #                              fig_size=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None, text=False)
    # # CHIR rest
    #
    # # day5
    # sp_tp = []  # day5
    # for i in range(11, 19):
    #     sp_tp.append('2018-12-06~II-2_CD13~T' + str(i))
    # merge_specific_time_point_features(main_path, features_path, sp_tp, 'features_Day5.csv')
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\features_Day5.csv'
    # do_pca(main_path, input_csv_path, 'PCA_vertical_Day5.csv', MAX_mle=3)
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\PCA_vertical_Day5.csv'
    # pca_vertical_to_horizontal(main_path, input_csv_vertical, 'PCA_horizontal_Day5.csv')
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\PCA_horizontal_Day5.csv'
    # do_draw_dot_flow_3point_CD13(main_path, input_csv_vertical, input_csv_horizontal, IF_file, 'PCA_Day5',
    #                              fig_size=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None, text=False)
    # # day5
    #
    # # day6
    # sp_tp = []  # day6
    # for i in range(1, 14):
    #     sp_tp.append('2018-12-07~II-3_CD13~T' + str(i))
    # merge_specific_time_point_features(main_path, features_path, sp_tp, 'features_Day6.csv')
    # input_csv_path = r'C:\Users\Kitty\Desktop\CD13\features_Day6.csv'
    # do_pca(main_path, input_csv_path, 'PCA_vertical_Day6.csv', MAX_mle=3)
    # input_csv_vertical = r'C:\Users\Kitty\Desktop\CD13\PCA_vertical_Day6.csv'
    # pca_vertical_to_horizontal(main_path, input_csv_vertical, 'PCA_horizontal_Day6.csv')
    # input_csv_horizontal = r'C:\Users\Kitty\Desktop\CD13\PCA_horizontal_Day6.csv'
    # do_draw_dot_flow_3point_CD13(main_path, input_csv_vertical, input_csv_horizontal, IF_file, 'PCA_Day6',
    #                              fig_size=(12.80, 10.24), x_min=None, x_max=None, y_min=None, y_max=None, text=False)
    # # day6
