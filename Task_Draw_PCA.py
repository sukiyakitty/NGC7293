import os
import sys
import math
import time
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.decomposition import PCA
from Lib_Function import get_specific_image, is_number, is_float
from Lib_Manifold import cacu_save_pca, cacu_draw_save_pca


def main(args):
    # ---the pandas display settings---
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', 100)
    # pd.set_option('max_colwidth', 100)
    # pd.set_option('display.width', None)

    global main_path, MAX_mle, step
    main_path = args.main_path
    MAX_mle = 3
    step = args.step
    print('The step is:', step)

    all_features = pd.read_csv(os.path.join(main_path, 'Features.csv'), header=0, index_col=0)
    all_features = all_features.applymap(is_number)
    all_features = all_features.dropna(axis=0, how='any')
    print(all_features)

    main_pca = PCA(n_components=MAX_mle)
    all_counts = all_features.shape[0]

    if not args.exp_finished:
        print('The Experiment is ongoing...')
        j = 1
        while True:
            all_features = pd.read_csv(os.path.join(main_path, 'Features.csv'), header=0, index_col=0)
            all_features = all_features.applymap(is_number)
            all_features = all_features.dropna(axis=0, how='any')
            if all_features.shape[0] >= j * step:
                pca_result = main_pca.fit_transform(all_features.iloc[0:j * step])
                print(main_pca.explained_variance_ratio_)
                fig = plt.figure()
                plt.scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])
                fig.savefig(os.path.join(main_path, 'figure', str(pca_result.shape[0]) + '.png'))
                plt.close()
                pca_result_DF = pd.DataFrame(pca_result, index=all_features.index,
                                             columns=['pca' + str(col) for col in range(1, MAX_mle + 1)])
                pca_result_DF.to_csv(path_or_buf=os.path.join(main_path, 'PCA.csv'))
                j += 1
            else:
                time.sleep(0.618)
            if False:
                # always waiting for features comming
                break
    else:  # elif args.exp_finished == True:
        print('The Experiment had finished!')
        for i in range(step, all_counts + 1, step):
            pca_result = main_pca.fit_transform(all_features.iloc[0:i])
            print(main_pca.explained_variance_ratio_)
            fig = plt.figure()
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])
            fig.savefig(os.path.join(main_path, 'figure', str(pca_result.shape[0]) + '.png'))
            plt.close()
        pca_result = main_pca.fit_transform(all_features)
        print(main_pca.explained_variance_ratio_)
        pca_result_DF = pd.DataFrame(pca_result, index=all_features.index,
                                     columns=['pca' + str(col) for col in range(1, MAX_mle + 1)])
        pca_result_DF.to_csv(path_or_buf=os.path.join(main_path, 'PCA.csv'))
    # else:
    #     print('Whether experiment finished? 0:not finished; 1:yes,the experiment finished! ')
    #     exit('Exit...')


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', type=str, nargs='?', default=r'C:\Users\Kitty\Desktop\test',
                        help='The Main Folder Path')
    parser.add_argument('--exp_finished', type=bool, nargs='?', default=True,
                        help='whether experiment finished? 0:not finished; 1:yes,the experiment finished! ')
    parser.add_argument('--step', type=int, nargs='?', default=10, help='The PCA image step')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
