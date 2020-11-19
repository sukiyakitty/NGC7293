import sys
import os
import argparse
import pandas as pd
from Lib_Class import ImageData
from Lib_Function import get_specific_image


def basic_task_density_extraction(args_path, this_F_index, this_B, this_T, this_S, this_Z, this_C, this_M):
    analysis_data_mem = pd.DataFrame(
        columns=['batch', 'date', 'name', 'index_B', 'index_T', 'index_S', 'index_Z', 'index_C', 'index_M',
                 'is_benchmark', 'is_discard', 'analysis_method', 'density', 'features', 'pca'])
    # index is path_date_name_BTSZCM_image
    # all_features = pd.DataFrame(columns=['f' + str(col) for col in range(1, 257)])
    # index is path_date_name_BTSZCM_image
    # global args_path, cardinal_mem, analysis_data_mem, all_features
    this_image_path = get_specific_image(this_F_index, this_B, this_T, this_S, this_Z, this_C, this_M)  # Z=2; C=1;
    if this_image_path is not None and os.path.exists(this_image_path):
        this_image = ImageData(this_image_path, 0)
        # getSift
        # all_features.loc[this_image_path] = this_image.getSift()
        # getSift
        analysis_data_mem.loc[
            this_image_path, ['index_B', 'index_T', 'index_S', 'index_Z', 'index_C', 'index_M',
                              'is_benchmark', 'is_discard', 'analysis_method', 'density', 'features', 'pca']] = [
            this_B, this_T, this_S, this_Z, this_C, this_M, 0, 0, 1, this_image.density, 0, 0]
        # cardinal_mem.loc[this_F_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = [this_B, this_T,
        #                                                                                           this_S, this_Z,
        #                                                                                           this_C, this_M]
        # if not all_features.empty:
        #     all_features.to_csv(path_or_buf=os.path.join(args_path, 'Features.csv'), mode='a', header=None)
        if not analysis_data_mem.empty:
            analysis_data_mem.to_csv(path_or_buf=os.path.join(args_path, 'Analysis_Data.csv'), mode='a', header=None)
        # cardinal_mem.to_csv(path_or_buf=os.path.join(args_path, 'Cardinal.csv'))
        print('  !!!success!!!  ', end='')
    else:
        print('  !!!ZEN Export error!!!  ', end='')
        return False
    return this_image.density


def basic_task_feature_extraction(args_path, this_F_index, this_B, this_T, this_S, this_Z, this_C, this_M):
    analysis_data_mem = pd.DataFrame(
        columns=['batch', 'date', 'name', 'index_B', 'index_T', 'index_S', 'index_Z', 'index_C', 'index_M',
                 'is_benchmark', 'is_discard', 'analysis_method', 'density', 'features', 'pca'])
    # index is path_date_name_BTSZCM_image
    all_features = pd.DataFrame(columns=['f' + str(col) for col in range(1, 257)])
    # index is path_date_name_BTSZCM_image
    # global args_path, cardinal_mem, analysis_data_mem, all_features
    this_image_path = get_specific_image(this_F_index, this_B, this_T, this_S, this_Z, this_C, this_M)  # Z=2; C=1;
    if this_image_path is not None and os.path.exists(this_image_path):
        this_image = ImageData(this_image_path, 0)
        # getSift
        all_features.loc[this_image_path] = this_image.getSIFT()
        # getSift
        analysis_data_mem.loc[
            this_image_path, ['index_B', 'index_T', 'index_S', 'index_Z', 'index_C', 'index_M',
                              'is_benchmark', 'is_discard', 'analysis_method', 'density', 'features', 'pca']] = [
            this_B, this_T, this_S, this_Z, this_C, this_M, 0, 0, 1, this_image.density, 1, 0]
        # cardinal_mem.loc[this_F_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = [this_B, this_T,
        #                                                                                           this_S, this_Z,
        #                                                                                           this_C, this_M]
        if not all_features.empty:
            all_features.to_csv(path_or_buf=os.path.join(args_path, 'Features.csv'), mode='a', header=None)
        if not analysis_data_mem.empty:
            analysis_data_mem.to_csv(path_or_buf=os.path.join(args_path, 'Analysis_Data.csv'), mode='a', header=None)
        # cardinal_mem.to_csv(path_or_buf=os.path.join(args_path, 'Cardinal.csv'))
        print('  !!!success!!!  ', end='')
    else:
        print('  !!!ZEN Export error!!!  ', end='')
        return False
    return True


def main(args):
    # this program can extraction cell density(1) and SIFT features(256) of one M
    print('<<<---', args.this_path, '  ', args.this_B, args.this_T, args.this_S, args.this_Z, args.this_C, args.this_M,
          '--->>>')
    basic_task_feature_extraction(args.main_path, args.this_path, args.this_B, args.this_T, args.this_S, args.this_Z,
                                  args.this_C, args.this_M)


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', type=str, nargs='?', default=r'D:\CD11_small', help='The Main Folder Path')
    parser.add_argument('--this_path', type=str, nargs='?',
                        default=r'D:\CD11_small\2018-11-08\2018-11-08_1100_I-1_CD11',
                        help='The output Image File Folder Path')
    parser.add_argument('--this_B', type=int, nargs='?', default=1, help='this Block')
    parser.add_argument('--this_T', type=int, nargs='?', default=7, help='this Time')
    parser.add_argument('--this_S', type=int, nargs='?', default=52, help='this Scene')
    parser.add_argument('--this_Z', type=int, nargs='?', default=2, help='this Z-direction')
    parser.add_argument('--this_C', type=int, nargs='?', default=1, help='this Channel')
    parser.add_argument('--this_M', type=int, nargs='?', default=7, help='this Mosaic tiles')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
