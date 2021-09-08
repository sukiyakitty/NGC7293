import os
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
from Lib_Function import is_number


def cacu_save_pca(main_path):
    max_mle = 3
    all_features = pd.read_csv(os.path.join(main_path, 'Features.csv'), header=0, index_col=0)
    all_features = all_features.applymap(is_number)
    all_features = all_features.dropna(axis=0, how='any')
    main_pca = PCA(n_components=max_mle)
    pca_result = main_pca.fit_transform(all_features)
    # print('PCA variance ratio:', main_pca.explained_variance_ratio_)
    pca_result_DF = pd.DataFrame(pca_result, index=all_features.index,
                                 columns=['pca' + str(col) for col in range(1, max_mle + 1)])
    pca_result_DF.to_csv(path_or_buf=os.path.join(main_path, 'PCA.csv'))
    return main_pca.explained_variance_ratio_


def cacu_draw_save_pca(main_path):
    MAX_mle = 3
    all_features = pd.read_csv(os.path.join(main_path, 'Features.csv'), header=0, index_col=0)
    all_features = all_features.applymap(is_number)
    all_features = all_features.dropna(axis=0, how='any')
    main_pca = PCA(n_components=MAX_mle)
    pca_result = main_pca.fit_transform(all_features)
    # print('PCA variance ratio:', main_pca.explained_variance_ratio_)
    pca_result_DF = pd.DataFrame(pca_result, index=all_features.index,
                                 columns=['pca' + str(col) for col in range(1, MAX_mle + 1)])
    pca_result_DF.to_csv(path_or_buf=os.path.join(main_path, 'PCA.csv'))
    fig = plt.figure()
    # plt.plot(pca_result[:, 0], pca_result[:, 1], '.')
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])
    # plt.show()
    # fig.show()
    fig.savefig(os.path.join(main_path, 'figure', str(pca_result.shape[0]) + '.png'))
    plt.close()
    return main_pca.explained_variance_ratio_


def cacu_draw3D_save_pca(main_path):
    MAX_mle = 3
    all_features = pd.read_csv(os.path.join(main_path, 'Features.csv'), header=0, index_col=0)
    all_features = all_features.applymap(is_number)
    all_features = all_features.dropna(axis=0, how='any')
    main_pca = PCA(n_components=MAX_mle)
    pca_result = main_pca.fit_transform(all_features)
    # print('PCA variance ratio:', main_pca.explained_variance_ratio_)
    pca_result_DF = pd.DataFrame(pca_result, index=all_features.index,
                                 columns=['pca' + str(col) for col in range(1, MAX_mle + 1)])
    pca_result_DF.to_csv(path_or_buf=os.path.join(main_path, 'PCA.csv'))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], '.')
    # ax.scatter3D(pca_result[:,0],pca_result[:,1],pca_result[:,2])
    # plt.show()
    # fig.show()
    fig.savefig(os.path.join(main_path, 'figure', str(pca_result.shape[0]) + '_3D.png'))
    plt.close()
    return main_pca.explained_variance_ratio_


def do_pca(main_path, input_csv_path, output_csv, MAX_mle=3, draw_save=False, draw=False, figsize=(128.0, 102.4), D=2,
           shape='-', x_min=None, x_max=None, y_min=None, y_max=None):
    # input one csv feature and do once pca analysis output a pca csv file
    # if draw_save,
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_path:  r'C:\Users\Kitty\Desktop\CD13\All_FEATURES.csv'
    # output_csv: 'All_DATA_PCA.csv'
    # MAX_mle = pca numbers
    # draw_save=True : save the pca visualization result to .png
    # draw=False : do draw the pca visualization on screen? (NOTICE if draw_save=False, always do not draw)
    # D=2 : pca visualization dimension
    # shape='-' : the matplotlib plot shape '-' is line ; '.' is dot
    # x_min=None, x_max=None, y_min=None, y_max=None : the pca picture x y axis limit: plt.xlim
    text = False

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_path):
        print('!ERROR! The input_csv_path does not existed!')
        return False

    all_data = pd.read_csv(input_csv_path, header=0, index_col=0)
    all_data = all_data.applymap(is_number)
    all_data = all_data.dropna(axis=0, how='any')

    all_pca = PCA(n_components=MAX_mle)
    pca_result = all_pca.fit_transform(all_data)
    # print('PCA variance ratio:', main_pca.explained_variance_ratio_)
    pca_result_DF = pd.DataFrame(pca_result, index=all_data.index,
                                 columns=['pca' + str(col) for col in range(1, MAX_mle + 1)])
    pca_result_DF.to_csv(path_or_buf=os.path.join(main_path, output_csv))

    if draw_save:

        well_name_list = []
        for i_str in all_data.index:
            well_name_list.append(int(i_str.split('~')[0].split('S')[1]))
        well_name_S = pd.Series(well_name_list, name='S_name')
        well_count_S = well_name_S.groupby(well_name_S).count()
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
        for i in range(0, len(well_count_S)):
            i_range = range(i_index, i_index + well_count_S.values[i])
            i_index = i_index + well_count_S.values[i]

            c = range(1, well_count_S.values[i] + 1)
            # c = pca_result[i_range, 2]
            color = np.random.rand(3)
            if D == 2:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                if shape == '-':
                    plt.plot(pca_result[i_range, 0], pca_result[i_range, 1], color=color, linestyle='-',
                             label='S' + str(well_count_S.index[i]))
                    plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c)
                elif shape == '.':
                    plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c)
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

        plt.legend(loc='upper right')
        fig.savefig(os.path.join(main_path, output_csv + '.png'))

        if draw:
            plt.show()
        plt.close()

    return True


def do_manifold(main_path, features_csv, features_cols=None, output_folder='MainFold_ALL', n_neighbors=10,
                n_components=3):
    # input one feature.csv and do once manifold analysis output a manifold.csv file
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # features_csv:  r'C:\Users\Kitty\Desktop\CD13\All_FEATURES.csv' （row:elements;col:features;）
    # features_cols: features_cols range(0,256) (class:range)
    # output_folder: 'MainFold'
    # n_neighbors=10
    # n_components=3

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    features_csv = os.path.join(main_path, features_csv)
    if not os.path.exists(features_csv):
        print('!ERROR! The features_csv does not existed!')
        return False
    if os.path.exists(os.path.join(main_path, output_folder)):
        shutil.rmtree(os.path.join(main_path, output_folder))
    os.makedirs(os.path.join(main_path, output_folder))

    all_features = pd.read_csv(features_csv, header=0, index_col=0)
    all_features = all_features.applymap(is_number)
    all_features = all_features.dropna(axis=0, how='any')
    # n_points = all_features.shape[0]
    if features_cols is None:
        features_cols = range(0, all_features.shape[1])
    X = all_features.iloc[:, features_cols].values

    t0 = time.time()
    Pca = PCA(n_components=n_components)
    Y = Pca.fit_transform(X)
    Pca_ratio = Pca.explained_variance_ratio_
    Y_DF = pd.DataFrame(Y, index=all_features.index,
                        columns=['pca' + str(Pca_ratio[i]) for i in range(0, len(Pca_ratio))])
    Y_DF.to_csv(path_or_buf=os.path.join(main_path, output_folder, 'PCA.csv'))
    t1 = time.time()
    print("%s: %.2g sec" % ('PCA', t1 - t0))

    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian_LLE', 'Modified_LLE']

    methods = ['standard', 'modified']
    labels = ['LLE', 'Modified_LLE']

    for i, method in enumerate(methods):
        t0 = time.time()
        Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
                                            method=method).fit_transform(X)
        Y_DF = pd.DataFrame(Y, index=all_features.index, columns=[method + str(l) for l in range(1, n_components + 1)])
        Y_DF.to_csv(path_or_buf=os.path.join(main_path, output_folder, labels[i] + '.csv'))
        t1 = time.time()
        print("%s: %.2g sec" % (labels[i], t1 - t0))

    t0 = time.time()
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    Y_DF = pd.DataFrame(Y, index=all_features.index, columns=['Isomap' + str(l) for l in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(main_path, output_folder, 'Isomap.csv'))
    t1 = time.time()
    print("Isomap: %.2g sec" % (t1 - t0))

    t0 = time.time()
    Y = manifold.MDS(n_components, max_iter=100, n_init=1).fit_transform(X)
    Y_DF = pd.DataFrame(Y, index=all_features.index, columns=['MDS' + str(l) for l in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(main_path, output_folder, 'MDS.csv'))
    t1 = time.time()
    print("MDS: %.2g sec" % (t1 - t0))

    t0 = time.time()
    Y = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)
    Y_DF = pd.DataFrame(Y, index=all_features.index,
                        columns=['SpectralEmbedding' + str(l) for l in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(main_path, output_folder, 'SpectralEmbedding.csv'))
    t1 = time.time()
    print("SpectralEmbedding: %.2g sec" % (t1 - t0))

    t0 = time.time()
    Y = manifold.TSNE(n_components=n_components, init='pca', random_state=0).fit_transform(X)
    Y_DF = pd.DataFrame(Y, index=all_features.index, columns=['tSNE' + str(l) for l in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(main_path, output_folder, 'tSNE.csv'))
    t1 = time.time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    return True


def do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
                                n_components=3):
    # input a feature.csv list conbination and do manifold analysis output one manifold.csv file
    # features_csv_list:  [r'C:\Users\Kitty\Desktop\CD13\All_FEATURES.csv',r'C:\Users\Kitty\Desktop\CD26\All_FEATURES.csv' ]（row:elements;col:features;）
    # features_cols: features_cols range(0,256) (class:range) or None(all)
    # output_folder: a Summary folder
    # n_neighbors=10
    # n_components=3

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    if type(name_list) is list:
        if len(name_list) == 0:
            print('!ERROR! The name_list must have one str!')
            return False
        else:
            for name in name_list:
                if type(name) is str:
                    pass
                else:
                    print('!ERROR! The name must be str!')
                    return False
    elif type(name_list) is str:
        name_list = [name_list]
    else:
        print('!ERROR! The name_list must be str or str list!')
        return False

    if type(features_file_list) is list:
        if len(features_file_list) == 0:
            print('!ERROR! The features_file_list must have one file!')
            return False
        else:
            for file in features_file_list:
                if os.path.exists(file):
                    pass
                else:
                    print('!ERROR! The file must be exist!')
                    return False
    elif os.path.exists(features_file_list):
        features_file_list = [features_file_list]
    else:
        print('!ERROR! The features_file_list must be existed file or files list!')
        return False

    if len(name_list) != len(features_file_list):
        print('!ERROR! The name_list must paired with features_file_list!')
        return False

    i = 0
    for each_name in name_list:
        this_DF = pd.read_csv(features_file_list[i], header=0, index_col=0)
        this_index = this_DF.index
        this_index = each_name + '~' + this_index
        this_DF.index = this_index
        if i == 0:
            all_features_DF = this_DF
        else:
            all_features_DF = all_features_DF.append(this_DF)
        i += 1

    all_features_DF = all_features_DF.applymap(is_number)
    all_features_DF = all_features_DF.dropna(axis=0, how='any')

    # n_points = all_features.shape[0]
    if features_cols is None:
        features_cols = range(0, all_features_DF.shape[1])
    X = all_features_DF.iloc[:, features_cols].values

    t0 = time.time()
    Pca = PCA(n_components=n_components)
    Y = Pca.fit_transform(X)
    Pca_ratio = Pca.explained_variance_ratio_
    Y_DF = pd.DataFrame(Y, index=all_features_DF.index,
                        columns=['pca' + str(Pca_ratio[i]) for i in range(0, len(Pca_ratio))])
    Y_DF.to_csv(path_or_buf=os.path.join(output_path, 'PCA.csv'))
    t1 = time.time()
    print("%s: %.2g sec" % ('PCA', t1 - t0))

    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian_LLE', 'Modified_LLE']

    methods = ['standard', 'modified']
    labels = ['LLE', 'Modified_LLE']

    for i, method in enumerate(methods):
        t0 = time.time()
        Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
                                            method=method).fit_transform(X)
        Y_DF = pd.DataFrame(Y, index=all_features_DF.index,
                            columns=[method + str(l) for l in range(1, n_components + 1)])
        Y_DF.to_csv(path_or_buf=os.path.join(output_path, labels[i] + '.csv'))
        t1 = time.time()
        print("%s: %.2g sec" % (labels[i], t1 - t0))

    t0 = time.time()
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    Y_DF = pd.DataFrame(Y, index=all_features_DF.index, columns=['Isomap' + str(l) for l in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(output_path, 'Isomap.csv'))
    t1 = time.time()
    print("Isomap: %.2g sec" % (t1 - t0))

    t0 = time.time()
    Y = manifold.MDS(n_components, max_iter=100, n_init=1).fit_transform(X)
    Y_DF = pd.DataFrame(Y, index=all_features_DF.index, columns=['MDS' + str(l) for l in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(output_path, 'MDS.csv'))
    t1 = time.time()
    print("MDS: %.2g sec" % (t1 - t0))

    t0 = time.time()
    Y = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)
    Y_DF = pd.DataFrame(Y, index=all_features_DF.index,
                        columns=['SpectralEmbedding' + str(l) for l in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(output_path, 'SpectralEmbedding.csv'))
    t1 = time.time()
    print("SpectralEmbedding: %.2g sec" % (t1 - t0))

    t0 = time.time()
    Y = manifold.TSNE(n_components=n_components, init='pca', random_state=0).fit_transform(X)
    Y_DF = pd.DataFrame(Y, index=all_features_DF.index, columns=['tSNE' + str(l) for l in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(output_path, 'tSNE.csv'))
    t1 = time.time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    return True


def pca_vertical_to_horizontal(main_path, input_csv_path, output_file):
    # pca csv file layout: vertical to horizontal
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_path: is the PCA result 3col csv file (row is S and T)
    #           PCA1    PCA2    PCA3
    # S1~T1
    # S1~T2
    # S1~T3
    # ...
    # S2~T1
    # S2~T2
    # ...
    # !!!notice: each well may have different time points (because: Manually stop image acquisition prematurely)
    # output_file: is the horizontal arrangement
    #            T1                  T2                  T3                  ...
    #     PCA1  PCA2  PCA3    PCA1  PCA2  PCA3    PCA1  PCA2  PCA3    PCA1  PCA2  PCA3
    # S1
    # S2
    # S3
    # ...

    input_csv_path = os.path.join(main_path, input_csv_path)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_csv_path):
        print('!ERROR! The input_csv_path does not existed!')
        return False

    pca_result_DF = pd.read_csv(input_csv_path, header=0, index_col=0)
    new_data_DF = None

    well_name_list = []
    for i_str in pca_result_DF.index:
        well_name_list.append(int(i_str.split('~')[0].split('S')[1]))
    well_name_S = pd.Series(well_name_list, name='S_name')
    well_count_S = well_name_S.groupby(well_name_S).count()  # the Series: time point number of each well

    offset_list = [0] * len(well_count_S)  # except S1 well, the other well will miss the last time point cell image

    time_point = well_count_S.values[0]  # all shot time point, using S1 well: well_count_S.values[0]
    time_point_benchmark_index = None  # the S1 well index (Experiment name & time point)

    for i_T in range(0, time_point):  # i_T is this time point; for each time point
        this_T_data = pd.DataFrame()  # this_T_data is
        i_index = 0
        for i_S in range(0, len(well_count_S)):  # for each S well
            this_S_name = 'S' + str(well_count_S.index[i_S])  # well name exp:'S1'
            this_S_range = range(i_index, i_index + well_count_S.values[i_S])  # the iloc index range of pca_result_DF
            i_index += well_count_S.values[i_S]  # this start index auto update exp:0
            # this_S_range[i_T] : this pca index
            # offset_list[i_S] : this well offset
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
                row_temp_ndarray = pca_result_DF.values[this_S_range[i_T - offset_list[i_S]]]
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

    new_data_DF.to_csv(path_or_buf=os.path.join(main_path, output_file))
    return True


def pca_analysis_separatly(main_path, analysis_path, MAX_mle=3, pca_save=True, draw_save=True, draw=False, D=2,
                           shape='-', do_all_pca=True, text=False):
    # do pca analysis (NOTICE it will do pca analysis separatly among all wells)
    # input: main_path: main path;
    # analysis_path: the path contained features .cvs files
    # MAX_mle = pca numbers
    # pca_save=True : save the pca result?
    # draw_save=True : save the pca visualization result to .png
    # draw=False : do draw the pca visualization on screen? (NOTICE if draw_save=False, always do not draw)
    # D=2 : pca visualization dimension
    # shape='-' : the matplotlib plot shape '-' is line ; '.' is dot
    # do_all_pca=True : combine all features, and do pca once for all
    # text=False : print time point number text on pca image, from 1 to n (reference to features.csv files)
    # output:
    # pca_folder = 'Analysis_PCA'
    # draw_folder = 'PCAFigure_Independent'
    # (main_path, 'All_DATA.csv')
    # (main_path, 'All_DATA_PCA.csv')
    # (main_path, 'All_DATA_PCA.png')

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(analysis_path):
        print('!ERROR! The analysis_path does not existed!')
        return False

    if pca_save:
        pca_folder = 'Analysis_PCA'
        if os.path.exists(os.path.join(main_path, pca_folder)):
            shutil.rmtree(os.path.join(main_path, pca_folder))
        os.makedirs(os.path.join(main_path, pca_folder))

    if draw_save:
        draw_folder = 'PCAFigure_Independent'
        if os.path.exists(os.path.join(main_path, draw_folder)):
            shutil.rmtree(os.path.join(main_path, draw_folder))
        os.makedirs(os.path.join(main_path, draw_folder))

    if do_all_pca:
        all_data = None
        img_count_list = []

    analysis_CSVs = os.listdir(analysis_path)
    analysis_CSVs.sort(key=lambda x: int(x.split('.csv')[0].split('S')[1]))
    for ana_csv in analysis_CSVs:
        this_features = pd.read_csv(os.path.join(analysis_path, ana_csv), header=0, index_col=0)
        this_features = this_features.applymap(is_number)
        this_features = this_features.dropna(axis=0, how='any')
        my_index = this_features.index
        my_index = ana_csv[:-4] + '~' + my_index
        this_features.index = my_index

        if do_all_pca:
            img_count_list.append(this_features.shape[0])
            if all_data is None:
                all_data = this_features
            else:
                all_data = all_data.append(this_features)

        this_pca = PCA(n_components=MAX_mle)
        pca_result = this_pca.fit_transform(this_features)
        # print('PCA variance ratio:', main_pca.explained_variance_ratio_)
        pca_result_DF = pd.DataFrame(pca_result, index=this_features.index,
                                     columns=['pca' + str(col) for col in range(1, MAX_mle + 1)])
        if pca_save:
            pca_result_DF.to_csv(path_or_buf=os.path.join(main_path, pca_folder, ana_csv))

        if draw_save:
            fig = plt.figure(figsize=(12.80, 10.24))
            c = range(1, pca_result.shape[0] + 1)
            # c = pca_result[:, 2]
            color = np.random.rand(3)
            if D == 2:
                if shape == '-':
                    # plt.plot(pca_result[:, 0], pca_result[:, 1], color=color, marker='.', linestyle='-', label=ana_csv[:-4])
                    plt.plot(pca_result[:, 0], pca_result[:, 1], color=color, linestyle='-', label=ana_csv[:-4])
                    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=c)
                elif shape == '.':
                    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=c)
                else:
                    plt.plot(pca_result[:, 0], pca_result[:, 1], color=color, marker=shape, label=ana_csv[:-4])
                if text:
                    for i in range(0, pca_result.shape[0]):
                        plt.text(pca_result[i, 0], pca_result[i, 1], str(i + 1))
            elif D == 3:
                ax = plt.axes(projection='3d')
                ax.plot3D([pca_result[:, 0]], [pca_result[:, 1]], [pca_result[:, 2]], shape, label=ana_csv[:-4])
                # ax.scatter3D(pca_result[:,0],pca_result[:,1],pca_result[:,2])
            else:
                print('!ERROR! The D does not support!')
                return False

            plt.legend(loc='upper right')
            fig.savefig(os.path.join(main_path, draw_folder, ana_csv[:-4] + '.png'))

            if draw:
                plt.show()
            plt.close()

    if do_all_pca:
        all_data.to_csv(path_or_buf=os.path.join(main_path, 'All_DATA.csv'))
        all_pca = PCA(n_components=MAX_mle)
        pca_result = all_pca.fit_transform(all_data)
        # print('PCA variance ratio:', main_pca.explained_variance_ratio_)
        pca_result_DF = pd.DataFrame(pca_result, index=all_data.index,
                                     columns=['pca' + str(col) for col in range(1, MAX_mle + 1)])
        if pca_save:
            pca_result_DF.to_csv(path_or_buf=os.path.join(main_path, 'All_DATA_PCA.csv'))

        if draw_save:
            fig = plt.figure(figsize=(128.0, 102.4))
            i_index = 0
            for i in range(0, len(analysis_CSVs)):
                i_range = range(i_index, i_index + img_count_list[i])
                i_index = i_index + img_count_list[i]

                c = range(1, img_count_list[i] + 1)
                # c = pca_result[i_range, 2]

                color = np.random.rand(3)
                if D == 2:
                    if shape == '-':
                        # plt.plot(pca_result[:, 0], pca_result[:, 1], color=color, marker='.', linestyle='-', label=ana_csv[:-4])
                        plt.plot(pca_result[i_range, 0], pca_result[i_range, 1], color=color, linestyle='-',
                                 label=analysis_CSVs[i][:-4])
                        plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c)
                    elif shape == '.':
                        plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c)
                    else:
                        plt.plot(pca_result[i_range, 0], pca_result[i_range, 1], color=color, marker=shape,
                                 label=analysis_CSVs[i][:-4])
                    if text:
                        for j in range(0, img_count_list[i]):
                            k = i_range[j]
                            plt.text(pca_result[k, 0], pca_result[k, 1], str(j + 1))
                elif D == 3:
                    ax = plt.axes(projection='3d')
                    ax.plot3D(pca_result[i_range, 0], pca_result[i_range, 1], pca_result[i_range, 2], shape,
                              label=analysis_CSVs[i][:-4])
                    # ax.scatter3D(pca_result[:,0],pca_result[:,1],pca_result[:,2])
                else:
                    print('!ERROR! The D does not support!')
                    return False

            plt.legend(loc='upper right')
            fig.savefig(os.path.join(main_path, 'All_DATA_PCA.png'))

            if draw:
                plt.show()
            plt.close()

    return True


def do_tSNE_and_draw(main_path, features_csv, output_folder, n_components=3, T=False):
    # input one feature.csv and do once manifold analysis output a manifold.csv file
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # features_csv:  r'C:\Users\Kitty\Desktop\CD13\All_FEATURES.csv' （row:elements;col:features;）
    # features_cols: features_cols range(0,256) (class:range)
    # output_folder: 'MainFold'
    # n_neighbors=10
    # n_components=3

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    features_csv = os.path.join(main_path, features_csv)
    if not os.path.exists(features_csv):
        print('!ERROR! The features_csv does not existed!')
        return False
    if not os.path.exists(os.path.join(main_path, output_folder)):
        # shutil.rmtree(os.path.join(main_path, output_folder))
        os.makedirs(os.path.join(main_path, output_folder))

    all_features = pd.read_csv(features_csv, header=0, index_col=0)
    # all_features = all_features.applymap(is_number)
    all_features = all_features.dropna(axis=0, how='any')
    all_features = all_features.ix[~(all_features == 0).all(axis=1), :]
    all_features = all_features.ix[:, ~(all_features == 0).all(axis=0)]
    if T:
        all_features = pd.DataFrame(all_features.values.T, index=all_features.columns, columns=all_features.index)

    n_points = all_features.shape
    print(n_points)

    # X = all_features.iloc[:, features_cols].values
    X = all_features.values

    t0 = time.time()
    Y_pca = PCA(n_components=n_components).fit_transform(X)
    Y_DF = pd.DataFrame(Y_pca, index=all_features.index,
                        columns=['pca' + str(col) for col in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(main_path, output_folder, features_csv[:-4] + '_PCA.csv'))
    t1 = time.time()
    print("%s: %.2g sec" % ('PCA', t1 - t0))

    t0 = time.time()
    Y_tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0).fit_transform(X)
    Y_DF = pd.DataFrame(Y_tsne, index=all_features.index, columns=['tSNE' + str(l) for l in range(1, n_components + 1)])
    Y_DF.to_csv(path_or_buf=os.path.join(main_path, output_folder, features_csv[:-4] + '_tSNE.csv'))
    t1 = time.time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    color = np.random.rand(3)
    D = 2
    shape = '.'
    text = False
    k = 0
    for Y in [Y_pca, Y_tsne]:
        k = k + 1
        c = range(1, Y.shape[0] + 1)
        fig = plt.figure(figsize=(12.80, 10.24))
        if D == 2:
            if shape == '-':
                # plt.plot(pca_result[:, 0], pca_result[:, 1], color=color, marker='.', linestyle='-', label=ana_csv[:-4])
                plt.plot(Y[:, 0], Y[:, 1], color=color, linestyle='-', label=features_csv[:-4])
                plt.scatter(Y[:, 0], Y[:, 1], c=c)
            elif shape == '.':
                plt.scatter(Y[:, 0], Y[:, 1], c=c)
            else:
                plt.plot(Y[:, 0], Y[:, 1], color=color, marker=shape, label=features_csv[:-4])
            if text:
                for i in range(0, Y.shape[0]):
                    plt.text(Y[i, 0], Y[i, 1], str(i + 1))
        elif D == 3:
            ax = plt.axes(projection='3d')
            ax.plot3D([Y[:, 0]], [Y[:, 1]], [Y[:, 2]], shape, label=features_csv[:-4])
            # ax.scatter3D(pca_result[:,0],pca_result[:,1],pca_result[:,2])
        else:
            print('!ERROR! The D does not support!')
            return False

        plt.legend(loc='upper right')
        plt.colorbar()
        fig.savefig(os.path.join(main_path, output_folder, features_csv[:-4] + '_' + str(k) + '_.png'))
        plt.close()

    return True


def temp(main_path, in_csv):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    in_csv = os.path.join(main_path, in_csv)
    if not os.path.exists(in_csv):
        print('!ERROR! The in_csv does not existed!')
        return False

    in_pd = pd.read_csv(in_csv, header=0, index_col=0)
    n_points = in_pd.shape
    print(n_points)

    Y = in_pd.values

    fig = plt.figure(figsize=(12.80, 10.24))
    c = range(1, Y.shape[0] + 1)
    color = np.random.rand(3)
    D = 2
    shape = '.'
    text = False
    if D == 2:
        if shape == '-':
            # plt.plot(pca_result[:, 0], pca_result[:, 1], color=color, marker='.', linestyle='-', label=ana_csv[:-4])
            plt.plot(Y[:, 0], Y[:, 1], color=color, linestyle='-', label=in_csv[:-4])
            plt.scatter(Y[:, 0], Y[:, 1], c=c)
        elif shape == '.':
            plt.scatter(Y[:, 0], Y[:, 1], c=c)
        else:
            plt.plot(Y[:, 0], Y[:, 1], color=color, marker=shape, label=in_csv[:-4])
        if text:
            for i in range(0, Y.shape[0]):
                plt.text(Y[i, 0], Y[i, 1], str(i + 1))
    elif D == 3:
        ax = plt.axes(projection='3d')
        ax.plot3D([Y[:, 0]], [Y[:, 1]], [Y[:, 2]], shape, label=in_csv[:-4])
        # ax.scatter3D(pca_result[:,0],pca_result[:,1],pca_result[:,2])
    else:
        print('!ERROR! The D does not support!')
        return False

    plt.legend(loc='upper right')
    plt.colorbar()
    fig.savefig(os.path.join(main_path, in_csv[:-4] + '.png'))

    draw = False
    if draw:
        plt.show()
    plt.close()

    return True


def do_pca_excel(main_path, input_excel_path, output_excel, in_sheet_name=0, MAX_mle=3, draw_save=False, draw=False,
                 figsize=(128.0, 102.4), D=2, shape='-', x_min=None, x_max=None, y_min=None, y_max=None, text=False):
    # input one csv feature and do once pca analysis output a pca csv file
    # if draw_save,
    # main_path: r'C:\Users\Kitty\Desktop\CD13'
    # input_csv_path:  r'C:\Users\Kitty\Desktop\CD13\All_FEATURES.csv'
    # output_csv: 'All_DATA_PCA.csv'
    # MAX_mle = pca numbers
    # draw_save=True : save the pca visualization result to .png
    # draw=False : do draw the pca visualization on screen? (NOTICE if draw_save=False, always do not draw)
    # D=2 : pca visualization dimension
    # shape='-' : the matplotlib plot shape '-' is line ; '.' is dot
    # x_min=None, x_max=None, y_min=None, y_max=None : the pca picture x y axis limit: plt.xlim

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(input_excel_path):
        print('!ERROR! The input_excel_path does not existed!')
        return False

    all_data = pd.read_excel(input_excel_path, sheet_name=in_sheet_name, header=0, index_col=0)
    all_data = all_data.applymap(is_number)
    all_data = all_data.dropna(axis=0, how='any')

    all_pca = PCA(n_components=MAX_mle)
    pca_result = all_pca.fit_transform(all_data)
    # print('PCA variance ratio:', main_pca.explained_variance_ratio_)
    pca_result_DF = pd.DataFrame(pca_result, index=all_data.index,
                                 columns=['pca' + str(col) for col in range(1, MAX_mle + 1)])
    pca_result_DF.to_excel(os.path.join(main_path, output_excel), sheet_name=in_sheet_name)

    if draw_save:

        well_name_list = []
        for i_str in all_data.index:
            well_name_list.append(int(i_str.split('~')[0].split('S')[1]))
        well_name_S = pd.Series(well_name_list, name='S_name')
        well_count_S = well_name_S.groupby(well_name_S).count()
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
        for i in range(0, len(well_count_S)):
            i_range = range(i_index, i_index + well_count_S.values[i])
            i_index = i_index + well_count_S.values[i]

            c = range(1, well_count_S.values[i] + 1)
            # c = pca_result[i_range, 2]
            color = np.random.rand(3)
            if D == 2:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                if shape == '-':
                    plt.plot(pca_result[i_range, 0], pca_result[i_range, 1], color=color, linestyle='-',
                             label='S' + str(well_count_S.index[i]))
                    plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c)
                elif shape == '.':
                    plt.scatter(pca_result[i_range, 0], pca_result[i_range, 1], c=c)
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

        plt.legend(loc='upper right')
        fig.savefig(os.path.join(main_path, output_excel + '.png'))

        if draw:
            plt.show()
        plt.close()

    return True


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Lib_Mainfold.py !')

    name_list = ['CD13', 'CD26']
    features_file_list = [r'E:\Image_Processing\CD13\diff_vector_Features3_fisrt10hours.csv',
                          r'E:\Image_Processing\CD26\diff_vector_Features3_fisrt10hours.csv']
    output_path = r'C:\Users\Kitty\Desktop\Multi_Batch'
    do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
                                n_components=3)

    # main_path = r'C:\Users\Kitty\Documents\Desktop\CD30\PROCESSING'
    # features_cols = range(4, 451)
    # features_csv = r'Classed_all_C8-60H_Features.csv'
    # output_folder = r'Classed_all_C8-60H_Features'
    # do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)
    # features_csv = r'Classed_all_C8-60H_MyPGC_Features.csv'
    # output_folder = r'Classed_all_C8-60H_MyPGC_Features'
    # do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)
    # features_csv = r'Classed_all_C8-60H_Enhanced_Features.csv'
    # output_folder = r'Classed_all_C8-60H_Enhanced_Features'
    # do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)

    # main_path = r'C:\Users\Kitty\Desktop\202005_XGY_RNA-Seq'
    # features_csv = r'new_o_1.csv'
    # features_csv = r'test.csv'
    # output_folder = r'C:\Users\Kitty\Desktop\202005_XGY_RNA-Seq'
    # do_tSNE_and_draw(main_path, features_csv, output_folder, n_components=3, T=True)
    # in_csv = r'GSE106118_UMI_count_merge_tSNE.csv'
    # temp(main_path, in_csv)

    # main_path = r'C:\C137\Sub_Projects\Time-lapse_living_cell_imaging_analysis\whole PCA\CD13_Test_20210812'
    # features_csv = r'All_Features.csv'
    # features_cols = range(384, 448)
    # output_folder = r'MainFold_ALL'
    # do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)
