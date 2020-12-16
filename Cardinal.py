import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import concurrent.futures as ccf
import multiprocessing as mp
from Lib_Class import CurrentFolderStructs, ImageData
from Lib_Function import scan_main_path, get_specific_image, get_cpu_python_process, scan_dish_margin
from Lib_Manifold import cacu_save_pca, cacu_draw_save_pca


def main(args):
    # time.sleep(30)
    print('Cardinal Version: 1.0')
    # ---the pandas display settings---
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.width', None)
    np.set_printoptions(threshold=None)

    # ---the main data structures---
    args_path = ''
    cardinal_mem = pd.DataFrame(
        columns=['scheduling_method', 'experiment_complete', 'exp_B', 'exp_T', 'exp_S', 'exp_Z', 'exp_C', 'exp_M',
                 'analysis_complete', 'ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M'])
    # index is path_date_name
    # 'scheduling_method': 0 no processing
    # 'scheduling_method': 1 row_method
    # 'scheduling_method': 2 col_method
    analysis_data_mem = pd.DataFrame(
        columns=['batch', 'date', 'name', 'index_B', 'index_T', 'index_S', 'index_Z', 'index_C', 'index_M',
                 'is_benchmark', 'is_discard', 'analysis_method', 'density', 'features', 'pca'])
    # index is path_date_name_BTSZCM_image
    # all_features = pd.DataFrame(columns=['f' + str(col) for col in range(1, 257)])
    # index is path_date_name_BTSZCM_image
    # pca_coordinate = pd.DataFrame(columns=['pca' + str(col) for col in range(1, 3 + 1)])
    # index is path_date_name_BTSZCM_image

    # --- other vars ---
    MAX_PYTHON_PROCESS = args.max_process
    TIME_SLICE = args.time_slice
    args_path = args.path
    zoom = args.zoom
    pca_sleep = 0

    # ---the args.path IS existed? ---
    print('Hello! Now this Program will do some Analysis of the Image that CD7 producing...')
    if os.path.exists(args.path):
        print('The Main Folder Path: ', args.path)
    else:
        exit('INPUT Folder Path does not existing!')
        return False

    # ---IS the experiment finished? ---
    if args.name == '':
        experiment_finished = True
    else:
        experiment_finished = False

    # ---the path_date_name structures---
    print('ALL the image files are here:')
    path_date_name = scan_main_path(args.path)  # notice that the scan_main_path can disappear
    this_number = 0
    for this_name in path_date_name:
        this_number += 1
        print('>>>NO.', this_number, '>>>', this_name)
        this_BTSCZM = CurrentFolderStructs(this_name)
        dog = True
        while dog:
            this_BTSCZM.re_init_structs()
            if this_BTSCZM.is_available:
                print('B=', this_BTSCZM.current_B, 'T=', this_BTSCZM.current_T, 'S=', this_BTSCZM.current_S, 'Z=',
                      this_BTSCZM.current_Z, 'C=', this_BTSCZM.current_C, 'M=', this_BTSCZM.current_M)
                cardinal_mem.loc[this_name] = {'scheduling_method': 0, 'experiment_complete': 0,
                                               'exp_B': this_BTSCZM.current_B,
                                               'exp_T': this_BTSCZM.current_T, 'exp_S': this_BTSCZM.current_S,
                                               'exp_Z': this_BTSCZM.current_Z, 'exp_C': this_BTSCZM.current_C,
                                               'exp_M': this_BTSCZM.current_M, 'analysis_complete': 0,
                                               'ana_B': 0, 'ana_T': 0, 'ana_S': 0, 'ana_Z': 0, 'ana_C': 0, 'ana_M': 0}
                dog = False
            else:
                time.sleep(0.1)

    # ---the cardinal.csv IS existed? it's structures? ---
    cardinal_old_csv = pd.DataFrame()
    if os.path.exists(os.path.join(args.path, 'Cardinal.csv')):
        cardinal_old_csv = pd.read_csv(os.path.join(args.path, 'Cardinal.csv'), header=0, index_col=0)
        cardinal_old_csv = cardinal_old_csv.fillna(0)
        cardinal_old_csv = cardinal_old_csv.applymap(lambda x: int(x))
    if cardinal_old_csv.empty:
        print('The First time to processing this Experiment!')
    else:
        print('Last time the processing STOP at here:')
        print(cardinal_old_csv)
        cardinal_mem[
            ['scheduling_method', 'experiment_complete', 'analysis_complete', 'ana_B', 'ana_T', 'ana_S', 'ana_Z',
             'ana_C', 'ana_M']] += cardinal_old_csv[
            ['scheduling_method', 'experiment_complete', 'analysis_complete', 'ana_B', 'ana_T', 'ana_S', 'ana_Z',
             'ana_C', 'ana_M']]
        cardinal_mem = cardinal_mem.fillna(0)
        cardinal_mem = cardinal_mem.applymap(lambda x: int(x))

    # ---IS the experiment finished? ---
    if experiment_finished:
        print('The experiment complete! NOT real-time Processing...')
        cardinal_mem['experiment_complete'] = 1
    else:
        this_F_index = os.path.join(args.path, args.date, args.name)
        print('The experiment is processing! Real-time Processing!!!')
        cardinal_mem['experiment_complete'] = 1
        cardinal_mem.loc[this_F_index, 'experiment_complete'] = 0
    cardinal_mem = cardinal_mem.fillna(0)
    cardinal_mem = cardinal_mem.applymap(lambda x: int(x))

    # ---copy last Cardinal_RTemp.csv to cardinal_mem---
    if os.path.exists(os.path.join(args.path, 'Cardinal_RTemp.csv')):
        cardinal_RTemp = pd.read_csv(os.path.join(args.path, 'Cardinal_RTemp.csv'), header=0, index_col=0)
        this_F_index = cardinal_RTemp.iloc[0].name
        cardinal_mem.loc[this_F_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = cardinal_RTemp.loc[
            this_F_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']]
        os.remove(os.path.join(args.path, 'Cardinal_RTemp.csv'))

    # ---cardinal_mem write to cardinal.csv ---
    cardinal_mem.to_csv(path_or_buf=os.path.join(args.path, 'Cardinal.csv'))
    print('The new structures is: ')
    print(cardinal_mem)

    # ---scheduling_method distribute ---
    # 'args.method': 0: Not do Features analysis!
    # 'args.method': 1: completed experiment using row method! (parallel execution)
    # 'args.method': 3: always col (parallel execution)
    # 'args.method': 4: always using col method and Sequential execution!
    # 'args.method': 5: simple Sequential execution!
    # 'scheduling_method': 0: no processing
    # 'scheduling_method': 1: row_method
    # 'scheduling_method': 2: col_method
    if args.method == 1:
        cardinal_mem.loc[
            (cardinal_mem['scheduling_method'] == 0) & (
                    cardinal_mem['experiment_complete'] == 1), 'scheduling_method'] = 1
        cardinal_mem.loc[
            (cardinal_mem['scheduling_method'] == 0) & (
                    cardinal_mem['experiment_complete'] == 0), 'scheduling_method'] = 2
    elif (args.method == 3) | (args.method == 4):
        finished_exp_count = \
            cardinal_mem.loc[(cardinal_mem['scheduling_method'] == 2) & (cardinal_mem['analysis_complete'] == 1)].shape[
                0]
        processing_exp_count = \
            cardinal_mem.loc[(cardinal_mem['scheduling_method'] == 2) & (cardinal_mem['analysis_complete'] == 0)].shape[
                0]
        unstart_exp_count = \
            cardinal_mem.loc[(cardinal_mem['scheduling_method'] == 0) & (cardinal_mem['analysis_complete'] == 0)].shape[
                0]
        error_exp_count = cardinal_mem.loc[cardinal_mem['scheduling_method'] == 1].shape[0]
        print('method 3/4 finished_exp_count:', finished_exp_count)
        print('method 3/4 processing_exp_count:', processing_exp_count)
        print('method 3/4 unstart_exp_count:', unstart_exp_count)
        print('method 3/4 error_exp_count:', error_exp_count)
        if processing_exp_count > 1 or error_exp_count > 0:
            print('Method 3/4 ERROR! Wrong using method 3/4!')
            exit('Method 3/4 ERROR! Wrong using method 3/4!')
            return False

    # ---the Analysis_Data.csv IS existed? or create it ---
    if os.path.exists(os.path.join(args.path, 'Analysis_Data.csv')):
        # analysis_data_mem = pd.read_csv(os.path.join(args.path, 'Analysis_Data.csv'), header=0, index_col=0)
        print('The Analysis_Data.csv is existed!')
    else:
        print('The Analysis_Data.csv does NOT existed!')
        analysis_data_mem.to_csv(path_or_buf=os.path.join(args.path, 'Analysis_Data.csv'))

    # # --- the Features.csv IS existed? or create it ---
    # if os.path.exists(os.path.join(args.path, 'Features.csv')):
    #     # all_features = pd.read_csv(os.path.join(args.path, 'Features.csv'), header=0, index_col=0)
    #     print('The Features.csv is existed!')
    # else:
    #     print('The Features.csv does NOT existed!')
    #     all_features.to_csv(path_or_buf=os.path.join(args.path, 'Features.csv'))

    # --- the PCA.csv IS existed? or create it ---
    # if os.path.exists(os.path.join(args.path, 'PCA.csv')):
    #     # pca_coordinate = pd.read_csv(os.path.join(args.path, 'PCA.csv'), header=0, index_col=0)
    #     pca_sleep = 0
    #     print('The PCA.csv is existed!')
    # else:
    #     pca_sleep = 30
    #     print('The PCA.csv does NOT existed!')
    #     pca_coordinate.to_csv(path_or_buf=os.path.join(args.path, 'PCA.csv'))
    #     # # do some PCA analysis

    # ---the AVG_Density.csv IS existed? or create it ---
    # if os.path.exists(os.path.join(args.path, 'AVG_Density.csv')):
    #     print('The AVG_Density.csv is existed!')
    #     # avg_density_mem = pd.read_csv(os.path.join(args.path, 'AVG_Density.csv'), header=0, index_col=0)
    # else:
    #     print('The AVG_Density.csv does NOT existed!')
    #     avg_density_mem = pd.DataFrame(columns=['S' + str(col) for col in range(1, args.S + 1)])
    #     avg_density_mem.to_csv(path_or_buf=os.path.join(args.path, 'AVG_Density.csv'))

    # ---the experiment plan Experiment_Plan.csv IS existed? (including the IPS_density, CHIR and time)---
    experiment_plan_csv = pd.DataFrame()
    if os.path.exists(os.path.join(args.path, 'Experiment_Plan.csv')):
        experiment_plan_csv = pd.read_csv(os.path.join(args.path, 'Experiment_Plan.csv'), header=0, index_col=0)
    if experiment_plan_csv.empty:
        print('The Experiment Plan is EMPTY!!!')
    else:
        print('The Experiment Plan is:')
        print(experiment_plan_csv)

    # ---the saved Benchmark IS existed? ---
    benchmark = pd.DataFrame()
    if os.path.exists(os.path.join(args.path, 'Benchmark.csv')):
        benchmark = pd.read_csv(os.path.join(args.path, 'Benchmark.csv'), header=0, index_col=0)
    if benchmark.empty:
        print('NO Benchmark file!!! De Novo Developmental Trajectory')
    else:
        print('the Benchmark Developmental Trajectory: ')
        # do some Benchmark PCA

    # ---this processing experiment ---
    if not experiment_finished:
        print('The experiment design :::')
        print('total           Block ::: ', args.B)
        print('total            Time ::: ', args.T)
        print('total           Scene ::: ', args.S)
        print('total     Z-direction ::: ', args.Z)
        print('total         Channel ::: ', args.C)
        print('total    Mosaic tiles ::: ', args.M)

    # --- What is the edge of the disc? ---
    tiles_margin = scan_dish_margin(args.path)
    print('The photograph\'s edge is :', tiles_margin)

    # --- make dir: figure to save PCA figures ---
    # if not os.path.exists(os.path.join(args.path, 'Figure')):
    #     os.makedirs(os.path.join(args.path, 'Figure'))
    # print('All the figures result is in : ', os.path.join(args.path, 'Figure'))

    # --- make dir: all the stitching images S01-S96 ---
    zoom_str = "%.0f%%" % (args.zoom * 100)
    # if not os.path.exists(os.path.join(args.path, 'SSS_' + zoom_str)):  # SSS stand for Sequential Stitching Scene
    #     os.makedirs(os.path.join(args.path, 'SSS_' + zoom_str))
    # for i in range(1, args.S + 1):
    #     if not os.path.exists(os.path.join(args.path, 'SSS_' + zoom_str, 'S' + str(i))):
    #         os.makedirs(os.path.join(args.path, 'SSS_' + zoom_str, 'S' + str(i)))
    print('Stitching Scene zoom is: ', zoom_str)
    print('Sequential Stitching Scene is in : ', os.path.join(args.path, 'SSS_' + zoom_str))

    # --- make dir: useful stitching images S01-S96 ---
    # if not os.path.exists(
    #         os.path.join(args.path, 'SSSS_' + zoom_str)):  # SSSS stand for Square Sequential Stitching Scene
    #     os.makedirs(os.path.join(args.path, 'SSSS_' + zoom_str))
    # for i in range(1, args.S + 1):
    #     if not os.path.exists(os.path.join(args.path, 'SSSS_' + zoom_str, 'S' + str(i))):
    #         os.makedirs(os.path.join(args.path, 'SSSS_' + zoom_str, 'S' + str(i)))
    print('Stitching Scene zoom is: ', zoom_str)
    print('Square Sequential Stitching Scene is in : ', os.path.join(args.path, 'SSSS_' + zoom_str))

    # ------ Initialization finished ------
    print('Initialization finished!', 'Now you are past the point of no return!')
    print('\n---------------------------------\n')

    # ------------ begin ------------
    # 'args.method': 0: Not do Features analysis!
    # 'args.method': 1: completed experiment using row method! (parallel execution)
    # 'args.method': 3: always col (parallel execution)
    # 'args.method': 4: always using col method and Sequential execution!
    # 'args.method': 5: simple Sequential execution!
    # col means: finish an experiment.czexp one by one, once it handel only one czexp
    # row means: do several experiment.czexp at one time, so once it handel several czexps
    if args.method == 0:
        print('Not do Features analysis!')
        print('Exit Features analysis ... ')
    elif args.method == 1:
        print('method 1 : completed experiment using row method! (parallel execution)')
        os.system(
            "start /min python Scheduling_2Directions.py --main_path {} --B {} --T {} --S {} --Z {} --C {} --M {} --rt_process {} --once_process {} --max_process {} --time_slice {}".format(
                args.path, args.B, args.T, args.S, args.Z, args.C, args.M, int(2 * MAX_PYTHON_PROCESS / 3),
                int(MAX_PYTHON_PROCESS / 3), int(2 * MAX_PYTHON_PROCESS / 3), TIME_SLICE))
    elif args.method == 2:
        print('method 2 : Nothing!')
    elif args.method == 3:
        print('method 3 : always using col method! (parallel execution) ')
        os.system(
            "start /min python Scheduling_Column.py --main_path {} --B {} --T {} --S {} --Z {} --C {} --M {} --max_process {} --time_slice {}".format(
                args.path, args.B, args.T, args.S, args.Z, args.C, args.M, MAX_PYTHON_PROCESS, TIME_SLICE))
    elif args.method == 4:
        print('method 4 : always using col method and Sequential execution!')
        os.system(
            "start /min python Scheduling_Sequential.py --main_path {} --B {} --T {} --S {} --Z {} --C {} --M {} --max_process {} --time_slice {} --zoom {} --overlap {} --silly {}".format(
                args.path, args.B, args.T, args.S, args.Z, args.C, args.M, MAX_PYTHON_PROCESS, TIME_SLICE, zoom,
                args.overlap, args.missing))
    elif args.method == 5:
        print('method 5 : simple Sequential execution!')
        os.system(
            "start /min python Scheduling_x.py --main_path {} --B {} --T {} --S {} --Z {} --C {} --M {} --max_process {} --time_slice {} --zoom {} --overlap {} --analysis {}".format(
                args.path, args.B, args.T, args.S, args.Z, args.C, args.M, MAX_PYTHON_PROCESS, TIME_SLICE, zoom,
                args.overlap, args.analysis))
    else:
        print('key method should be: 1:row Catching up or 3:always col!')
        print('Exit...')

    # ------ begin pca analysis ------
    if args.pca == 0:
        print('Not do PCA analysis!')
        print('Exit PCA analysis ...')
    elif args.pca == 1:
        print('PCA method 1 : Always analysis the last!')
        time.sleep(pca_sleep)
        watchdog_pca = True
        while watchdog_pca:
            print(cacu_draw_save_pca(args.path))
            time.sleep(15)
    elif args.pca == 2:
        print('PCA method 2 : Catching up way!')
        os.system("start python Task_Draw_PCA.py --main_path {} --exp_finished {} --step {}".format(args.path,
                                                                                                    experiment_finished,
                                                                                                    10))
    else:
        print('Whether do pca analysis? 0:Not; 1:Always analysis the last; 2: Catching up way')
        print('Exit...')

    # ------ begin others analysis ------
    time.sleep(30)


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, nargs='?', default=r'G:\CD53\Processing', help='The Main Folder Path')
    parser.add_argument('--date', type=str, nargs='?', default='2020-11-21',
                        help='ZEN automatic generated date folder')
    parser.add_argument('--name', type=str, nargs='?', default='CD53_IPS-1',
                        help='The output Image File folder Name, always the Experiment Name')
    parser.add_argument('--B', type=int, nargs='?', default=1, help='B=block')
    parser.add_argument('--T', type=int, nargs='?', default=-1, help='T=time')
    parser.add_argument('--S', type=int, nargs='?', default=96, help='S=scene')
    parser.add_argument('--Z', type=int, nargs='?', default=3, help='Z=z-direction')
    parser.add_argument('--C', type=int, nargs='?', default=1, help='C=channel')
    parser.add_argument('--M', type=int, nargs='?', default=25, help='M=mosaic tiles')
    parser.add_argument('--method', type=int, nargs='?', default=5,
                        help='key method 0: Not do Features analysis! 1: completed experiment using row method! (parallel execution) 3: always col (parallel execution) 4 : always using col method and Sequential execution! 5 : simple Sequential execution!')
    parser.add_argument('--pca', type=int, nargs='?', default=0,
                        help='Whether do pca analysis? 0:Not; 1:Always analysis the last; 2: Catching up way')
    parser.add_argument('--max_process', type=int, nargs='?', default=35, help='max_process')
    parser.add_argument('--time_slice', type=int, nargs='?', default=30, help='time_slice is the time waiting CD7')
    parser.add_argument('--zoom', type=float, nargs='?', default=1, help='Stitching whole image resize zoom')
    parser.add_argument('--overlap', type=float, nargs='?', default=0.05, help='Stitching overlap')
    parser.add_argument('--missing', type=int, nargs='?', default=0,
                        help='How to calculate avg_density & Stitching, if miss images?')
    parser.add_argument('--analysis', type=int, nargs='?', default=143, help='Analysis flag')
    # missing==0: Normal: If images missing DO NOT ( Calculate avg_density & Stitching )! ;
    # missing==1: Calculate avg_density ANYWAY! & If images missing DO NOT Stitching! ;
    # missing==2: Calculate avg_density ANYWAY! & DO NOT Stitching! ;
    # missing==3: Calculate avg_density ANYWAY! & Stitching ANYWAY! ;
    # --analysis：：：
    # 1bit:avg density;
    # 2bit:well density;
    # 3bit:core_analysis(); call function RT_PGC_Features(): alwayss do the SSS and SSSS my_PGC
    # 4bit:call_analysis(); new thread stiching wells
    # alwayss do the SSS and SSSS my_PGC
    # 5bit: SIFT\SURF\ORB;
    # 6bit: well_image Density;
    # 7bit: well_image Perimeter;
    # 8bit: Call Matlab Fractal Curving;
    # 10000111 = 135
    # 87654321 bit
    # 10000111 = 135
    # 10001111 = 143
    # 00001111 = 15

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
#     run Cardinal_old.py
#     start python Cardinal_old.py
#     python Cardinal_old.py C:\CD16 2019-01-16 II-1_CD16 1 -1 96 3 1 25
