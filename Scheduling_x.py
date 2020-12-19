import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from Lib_Class import CurrentFolderStructs
from Lib_Function import scan_dish_margin, get_specific_image, image_my_enhancement, get_AE_density, \
    get_img_density, saving_density_RT, saving_time, get_specific_Mpath, stitching_CZI, stitching_CZI_AutoBestZ, \
    get_cpu_python_process
from Lib_Features import old_core_features, core_features_enhanced, RT_PGC_Features
from Lib_Tiles import return_96well_25_Tiles, return_12well_324_Tiles

matrix_list = return_96well_25_Tiles()  # matrix_list
Density_Alarm = False  # if True then make a phone call when Density is density_threshold
density_threshold = 0.76  # density_threshold


def call_analysis(main_path, well_image):
    # call able analysis
    os.system(
        "start /min python Task_additional_analysis.py --main_path {} --well_image_0 {} --well_image_1 {}".format(main_path,
                                                                                                            well_image[
                                                                                                                0],
                                                                                                            well_image[
                                                                                                                1]))


def m3_scheduling(this_index, args_B, args_T, args_S, args_Z, args_C, args_M):
    global main_path, tiles_margin, cardinal_mem, image_file_type, zoom, overlap, do_analysis, MAX_PYTHON_process_number

    this_F_row = cardinal_mem.loc[this_index]
    # do_analysis = 0b0  # 1bit:avg density; 2bit:well density; 3bit:core_analysis(); 4bit:call_analysis()
    #                    # 5bit:do M analysis;
    if args_Z >= 1:
        Z = round((args_Z - 1) / 2 + 1)
        print('!Notice! USING z-direction: ', Z)
    else:
        print('!ERROR! z-direction must be an integer greater than zero!')
        return False
    if args_C >= 1:
        print('!Notice! USING channel 1')
        C = 1
    else:
        print('!ERROR! channel must be an integer greater than zero!')
        return False
    density_avg_base_list = []

    # B always 1 ! Different B, Different TSZCM, following do NOT Consider B:
    if this_F_row['ana_B'] == 0:
        this_start_loop_B = 1
    else:
        this_start_loop_B = this_F_row['ana_B']
    this_end_loop_B = this_F_row['exp_B']

    for this_B in range(this_start_loop_B, this_end_loop_B + 1):

        # T from this_start_loop_T to this_end_loop_T:
        this_start_loop_T = this_F_row['ana_T']
        if this_F_row['ana_T'] == 0:
            this_start_loop_T = 1
        this_end_loop_T = this_F_row['exp_T']

        for this_T in range(this_start_loop_T, this_end_loop_T + 1):

            # S from this_start_loop_S to this_end_loop_S:
            if this_T == this_F_row['ana_T']:
                this_start_loop_S = this_F_row[
                    'ana_S']  # very important: this S maybe not finished! this M must==args_M
            else:
                this_start_loop_S = 1
            if this_T == this_F_row['exp_T']:
                this_end_loop_S = this_F_row['exp_S']
            else:
                this_end_loop_S = args_S

            for this_S in range(this_start_loop_S, this_end_loop_S + 1):

                print('>>> Start ', this_index, ' ( T =', this_T, 'S =', this_S, ')')
                saving_time(main_path, 'Calculation_Time.csv', this_index, this_T, this_S)

                # Z? & C?
                for this_Z in range(1, args_Z + 1):
                    if this_Z == Z:

                        this_path = get_specific_Mpath(this_index, this_B, this_T, this_S, this_Z, C)
                        sum_pic_in_this_S = 0
                        sum_silly_S = 0
                        density_sum = 0
                        avg_density = -1

                        # M always from 1 to args_M
                        for this_M in range(1, args_M + 1):
                            this_image_path = get_specific_image(this_index, this_B, this_T, this_S, this_Z, C, this_M)
                            # print(this_image_path)
                            if this_image_path is not None and os.path.exists(this_image_path):
                                sum_pic_in_this_S += 1
                                if this_M not in tiles_margin:
                                    if do_analysis & 0b1 > 0:  # 1bit:avg density;
                                        sum_silly_S += 1
                                        density_sum += get_AE_density(this_index, this_B, this_T, this_S, this_Z, C,
                                                                      this_M)
                                    if do_analysis & 0b10000 > 0:  # 5bit:do M analysis;
                                        # do some M analysis
                                        pass
                            else:
                                print('!Notice! : Missing image!:', this_path, 'M:', this_M)

                        if do_analysis & 0b1 > 0:  # 1bit:avg density;
                            if sum_silly_S != 0:
                                avg_density = density_sum / sum_silly_S
                                print('The AVG_Density:', avg_density)
                                saving_density_RT(main_path, 'AVG_Density.csv', this_index, this_T, this_S, avg_density)
                            if avg_density >= 0 and Density_Alarm:
                                if len(density_avg_base_list) < 10:
                                    density_avg_base_list.append(avg_density)
                                else:
                                    density_avg_base_list.pop(0)
                                    density_avg_base_list.append(avg_density)
                                    density_avg_base = np.mean(density_avg_base_list)
                                if density_avg_base >= density_threshold:  # Convergence degree 75% Alarm!
                                    os.system(
                                        "start python Make_a_Call.py --phoneNum {} --content {}".format('17710805067',
                                                                                                        '4F60597DFF0C6211662F004300440037FF0C0049005000536C4754085EA68FBE5230767E52064E4B4E0353414E94FF0C8BF751C6590763626DB2'))
                        # !!! stitching
                        if sum_pic_in_this_S != 0:
                            print('!Key Step! : stitching_CZI_AutoBestZ( T =', this_T, 'S =', this_S, ')')
                            # stitching_result = square_stitching_save(main_path, this_index, this_B, this_T, this_S,
                            #                                          this_Z, C, args_M, square_side, zoom, overlap)
                            # stitching_result = stitching_CZI(main_path, this_index, this_B, this_T, this_S, this_Z, C,
                            #                                  matrix_list, zoom, overlap, output=None, do_SSSS=True)
                            stitching_result = stitching_CZI_AutoBestZ(main_path, this_index, this_B, this_T, this_S, C,
                                                                       matrix_list, zoom, overlap, output=None,
                                                                       do_SSSS=True, do_enhancement=False)
                        if sum_pic_in_this_S < args_M:
                            if this_S == this_end_loop_S:
                                print('!Notice! The latest unfinished Scene! Now continue...')
                                # return True
                            else:
                                print('!Warning! : ZEN auto export problems! Missing image!')
                                print('!Warning! :', this_path, 'only', sum_pic_in_this_S, 'imges had exported!')
                        elif sum_pic_in_this_S == args_M:
                            if do_analysis == 0:
                                pass
                            if do_analysis & 0b10 > 0:  # 2bit:well density;
                                zws_density = get_img_density(stitching_result[0])
                                znes_density = get_img_density(stitching_result[1])
                                print('The Whole_Well_Density:', zws_density)
                                print('The No_Edge_Density:', znes_density)
                                saving_density_RT(main_path, 'Whole_Well_Density.csv', this_index, this_T, this_S,
                                                  zws_density)
                                saving_density_RT(main_path, 'No_Edge_Density.csv', this_index, this_T, this_S,
                                                  znes_density)
                            if do_analysis & 0b100 > 0:  # 3bit:core_analysis();
                                print('!Notice! : Do in program analysis()!')
                                # old_core_features(main_path, stitching_result)
                                # core_features_enhanced(main_path, stitching_result)
                                RT_PGC_Features(main_path, stitching_result, analysis=do_analysis)
                                # features = 0b1101
                                # alwayss do the SSS and SSSS my_PGC
                                # 5bit: SIFT\SURF\ORB;
                                # 6bit: well_image Density;
                                # 7bit: well_image Perimeter;
                                # 8bit: Call Matlab Fractal Curving;
                            if do_analysis & 0b1000 > 0:  # 4bit:call_analysis();
                                print('!Notice! : Do some call_analysis()!')
                                call_analysis(main_path, stitching_result)

                        else:
                            print('!ERROR! Wrong M number !!!')
                            return False

                        cardinal_mem.loc[this_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = [this_B,
                                                                                                                this_T,
                                                                                                                this_S,
                                                                                                                this_Z,
                                                                                                                C,
                                                                                                                this_M]
                        cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))

                # S Multithreading control
                temp_p_number = get_cpu_python_process()
                print('!Notice! : $$$---Python Number: ', temp_p_number, '---$$$')
                if temp_p_number > MAX_PYTHON_process_number:
                    print('!Notice! : Python process number is more than ', MAX_PYTHON_process_number, ' Now sleep!')
                    time.sleep(2)
                    while True:
                        if get_cpu_python_process() >= MAX_PYTHON_process_number:
                            time.sleep(2)
                        else:
                            break

    cardinal_mem.loc[this_index, ['analysis_complete']] = 1
    cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
    return True


def m3_rt(this_index, args_B, args_T, args_S, args_Z, args_C, args_M):
    global main_path, tiles_margin, cardinal_mem, image_file_type, zoom, overlap, TIME_SLICE
    while True:
        time_start = time.time()

        if os.path.exists(os.path.join(this_index, 'ExperimentComplete.txt')):
            with open(os.path.join(this_index, 'ExperimentComplete.txt'), 'r') as fin:
                ec_str = fin.read()
            # os.remove(os.path.join(this_index, 'ExperimentComplete.txt'))
            cardinal_mem.loc[this_index, 'experiment_complete'] = 1

        this_cfs = CurrentFolderStructs(this_index)
        cardinal_mem.loc[this_index, ['exp_B', 'exp_T', 'exp_S', 'exp_Z', 'exp_C', 'exp_M']] = [
            this_cfs.current_B, this_cfs.current_T, this_cfs.current_S, this_cfs.current_Z, this_cfs.current_C,
            this_cfs.current_M]
        cardinal_mem.loc[this_index, ['analysis_complete']] = 0
        cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))

        m3_scheduling(this_index, args_B, args_T, args_S, args_Z, args_C, args_M)

        if cardinal_mem.loc[this_index, 'experiment_complete'] == 1 and cardinal_mem.loc[
            this_index, 'analysis_complete'] == 1:
            break

        time_duration = time.time() - time_start
        if cardinal_mem.loc[this_index, 'analysis_complete'] == 1 and time_duration < TIME_SLICE:
            print('!Notice! : Waiting CD7 shooting, Now sleep:', TIME_SLICE, 'S !')
            time.sleep(TIME_SLICE)

    return True


def main(args):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.width', None)

    global main_path, tiles_margin, cardinal_mem, image_file_type, zoom, overlap, TIME_SLICE, do_analysis, MAX_PYTHON_process_number

    main_path = args.main_path
    tiles_margin = scan_dish_margin(main_path)
    image_file_type = ('.jpg', '.png', '.tif')
    # zomm = args.zoom
    zoom = [1]  # be a float or a list, the first number is the processing pic zoom size,the rest is only store
    # square_side = 5
    overlap = args.overlap
    # Density_Alarm = 0
    MAX_PYTHON_process_number = args.max_process
    TIME_SLICE = args.time_slice
    do_analysis = args.analysis
    # matrix_list = return_96well_30_Tiles()

    cardinal_mem = pd.read_csv(os.path.join(main_path, 'Cardinal.csv'), header=0, index_col=0)
    cardinal_mem = cardinal_mem.fillna(0)
    cardinal_mem = cardinal_mem.applymap(lambda x: int(x))

    print('Hello I am the Scheduling_X.py program! Now I will sleep 5s!')
    time.sleep(5)
    print('Scheduling_Sequential.py is initialized! And the Cardinal.csv is:')
    print(cardinal_mem)

    for this_index, this_row in cardinal_mem.iterrows():
        if this_row['scheduling_method'] == 2 and this_row['experiment_complete'] == 1 and this_row[
            'analysis_complete'] == 1:
            print(this_index, 'Analysis Complete!!!')
            next
        elif this_row['scheduling_method'] == 2 and this_row['experiment_complete'] == 1 and this_row[
            'analysis_complete'] == 0:
            print(this_index, 'Catching up Analysising!!!')
            m3_scheduling(this_index, args.B, args.T, args.S, args.Z, args.C, args.M)
        elif this_row['scheduling_method'] == 2 and this_row['experiment_complete'] == 0:
            print(this_index, 'Real Time Analysising!!!')
            m3_rt(this_index, args.B, args.T, args.S, args.Z, args.C, args.M)
        elif this_row['scheduling_method'] == 0 and this_row['experiment_complete'] == 1 and this_row[
            'analysis_complete'] == 0:
            print(this_index, 'Waiting to Catching up!!!')
            cardinal_mem.loc[this_index, 'scheduling_method'] = 2
            cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
            m3_scheduling(this_index, args.B, args.T, args.S, args.Z, args.C, args.M)
        elif this_row['scheduling_method'] == 0 and this_row['experiment_complete'] == 0 and this_row[
            'analysis_complete'] == 0:
            print(this_index, 'Waiting to real time Analysising!!!')
            cardinal_mem.loc[this_index, 'scheduling_method'] = 2
            cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
            m3_rt(this_index, args.B, args.T, args.S, args.Z, args.C, args.M)
        else:
            print('Method 5 ERROR! Wrong using method 5!')
            time.sleep(30)
            exit('Method 5 ERROR! Wrong using method 5!')

    print('Mission Complete!!! Now return in 30 seconds!')
    time.sleep(30)


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', type=str, nargs='?',
                        default=r'G:\CD46\PROCESSING', help='The Main Folder Path')
    parser.add_argument('--B', type=int, nargs='?', default=1, help='B=block')
    parser.add_argument('--T', type=int, nargs='?', default=-1, help='T=time')
    parser.add_argument('--S', type=int, nargs='?', default=96, help='S=scene')
    parser.add_argument('--Z', type=int, nargs='?', default=3, help='Z=z-direction')
    parser.add_argument('--C', type=int, nargs='?', default=1, help='C=channel')
    parser.add_argument('--M', type=int, nargs='?', default=25, help='M=mosaic tiles')
    parser.add_argument('--max_process', type=int, nargs='?', default=35, help='max_process')
    parser.add_argument('--time_slice', type=int, nargs='?', default=30, help='time_slice')
    parser.add_argument('--zoom', type=float, nargs='?', default=1, help='Stitching whole image resize zoom')
    parser.add_argument('--overlap', type=float, nargs='?', default=0.05, help='Stitching overlap')
    parser.add_argument('--analysis', type=int, nargs='?', default=143, help='Analysis flag')
    # 1bit:avg density;
    # 2bit:well density;
    # 3bit:core_analysis(); call function RT_PGC_Features(): alwayss do the SSS and SSSS my_PGC
    # 4bit:call_analysis(); new thread stiching wells for a cultrue dish
    # 5bit: do M analysis pass; RT_PGC_Features(): SIFT\SURF\ORB;
    # 6bit: well_image Density;
    # 7bit: well_image Perimeter;bushbushi
    # 8bit: Call Matlab Fractal Curving;
    # 9bit: * After 15H Call Matlab Fractal Curving;
    # 10bit: ;
    # 11bit: ;
    # 12bit: ;
    # 210987654321   bit
    #     10000111 = 135
    #     10001111 = 143
    #         1111 = 15
    # 000100001111 = 271

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
