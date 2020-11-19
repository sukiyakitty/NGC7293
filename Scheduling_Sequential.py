import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from Lib_Class import CurrentFolderStructs
from Lib_Function import scan_dish_margin, get_specific_image, square_stitching_save, get_AE_density, \
    saving_scene_density_csv, get_img_density, saving_density_RT, saving_time


def m3_scheduling(this_index, args_B, args_T, args_S, args_Z, args_C, args_M):
    global main_path, tiles_margin, cardinal_mem, MAX_PYTHON_process_number, TIME_SLICE, image_file_type, square_side, zoom, overlap, silly, Density_Alarm, density_avg_base_list
    this_F_row = cardinal_mem.loc[this_index]
    execute_number = 0
    Z = 2
    C = 1
    if this_F_row['ana_B'] == 0:
        this_start_loop_B = 1
    else:
        this_start_loop_B = this_F_row['ana_B']
    this_end_loop_B = this_F_row['exp_B']
    # B always 1 ! Different B, Different TSZCM, following do NOT Consider B:
    for this_B in range(this_start_loop_B, this_end_loop_B + 1):
        this_start_loop_T = this_F_row['ana_T']
        if this_F_row['ana_T'] == 0:
            this_start_loop_T = 1
        this_end_loop_T = this_F_row['exp_T']
        for this_T in range(this_start_loop_T, this_end_loop_T + 1):
            if this_T == this_F_row['ana_T']:
                this_start_loop_S = this_F_row['ana_S']  # very important: this S maybe not finished! this M must==args_M
            else:
                this_start_loop_S = 1
            if this_T == this_F_row['exp_T']:
                this_end_loop_S = this_F_row['exp_S']
            else:
                this_end_loop_S = args_S
            for this_S in range(this_start_loop_S, this_end_loop_S + 1):
                # silly==0: Normal: If images missing DO NOT ( Calculate avg_density & Stitching )! ;
                # silly==1: Calculate avg_density ANYWAY! & If images missing DO NOT Stitching! ;
                # silly==2: Calculate avg_density ANYWAY! & DO NOT Stitching! ;
                # silly==3: Calculate avg_density ANYWAY! & Stitching ANYWAY! ;
                if silly == 3:  # silly==3: Calculate avg_density ANYWAY! & Stitching ANYWAY! ;
                    sum_pic_in_this_S = 0
                    sum_silly_S = 0
                    density_sum = 0
                    for this_M in range(1, args_M + 1):
                        this_image_path = get_specific_image(this_index, this_B, this_T, this_S, Z, C, this_M)
                        if this_image_path is not None and os.path.exists(this_image_path):
                            sum_pic_in_this_S += 1
                            if this_M not in tiles_margin:
                                sum_silly_S += 1
                                print('<<<--- Block:', this_B, 'Time:', this_T, 'Scene:', this_S, 'Mosaic Tiles:',
                                      this_M, end='')
                                density_sum += get_AE_density(this_index, this_B, this_T, this_S, Z, C, this_M)
                                print(' --->>>')
                        else:
                            print('Block:', this_B, 'Time:', this_T, 'Scene:', this_S,
                                  'has ZEN auto export problems! missing:', this_M)
                    if sum_silly_S != 0:
                        avg_density = density_sum / sum_silly_S
                        print('Silly: sum of Scene:', sum_silly_S, 'each images avg_density:', avg_density)
                        # saving_scene_density_csv(main_path, this_index, this_T, this_S, avg_density)
                        saving_density_RT(main_path, 'AVG_Density.csv', this_index, this_T, this_S, avg_density)
                    if sum_pic_in_this_S != 0:
                        stitching_result = square_stitching_save(main_path, this_index, this_B, this_T, this_S, Z, C,
                                                                 args_M, square_side, zoom, overlap)
                    if sum_pic_in_this_S == args_M:
                        zws_density = get_img_density(stitching_result[0])
                        znes_density = get_img_density(stitching_result[1])
                        print('zoom   whole stitching:', zws_density)
                        print('zoom no edge stitching:', znes_density)
                        # saving_scene_density_csv(main_path, this_index, this_T, this_S, avg_density)
                        saving_density_RT(main_path, 'Whole_Well_Density.csv', this_index, this_T, this_S, zws_density)
                        saving_density_RT(main_path, 'No_Edge_Density.csv', this_index, this_T, this_S, znes_density)
                    cardinal_mem.loc[this_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = [
                        this_B, this_T, this_S, Z, C, this_M]
                    cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
                elif silly == 2:  # silly==2: Calculate avg_density ANYWAY! & DO NOT Stitching! ;
                    sum_silly_S = 0
                    density_sum = 0
                    for this_M in range(1, args_M + 1):
                        if this_M not in tiles_margin:
                            this_image_path = get_specific_image(this_index, this_B, this_T, this_S, Z, C, this_M)
                            if this_image_path is not None and os.path.exists(this_image_path):
                                sum_silly_S += 1
                                print('<<<--- Block:', this_B, 'Time:', this_T, 'Scene:', this_S, 'Mosaic Tiles:',
                                      this_M,
                                      end='')
                                density_sum += get_AE_density(this_index, this_B, this_T, this_S, Z, C, this_M)
                                print(' --->>>')
                            else:
                                print('Block:', this_B, 'Time:', this_T, 'Scene:', this_S,
                                      'has ZEN auto export problems! missing:', this_M)
                    if sum_silly_S != 0:
                        print('Silly: sum of Scene:', sum_silly_S)
                        avg_density = density_sum / sum_silly_S
                        # saving_scene_density_csv(main_path, this_index, this_T, this_S, avg_density)
                        saving_density_RT(main_path, 'AVG_Density.csv', this_index, this_T, this_S, avg_density)
                    cardinal_mem.loc[this_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = [
                        this_B, this_T, this_S, Z, C, this_M]
                    cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
                elif silly == 1:  # silly==1: Calculate avg_density ANYWAY! & If images missing DO NOT Stitching! ;
                    sum_pic_in_this_S = 0
                    sum_silly_S = 0
                    density_sum = 0
                    for this_M in range(1, args_M + 1):
                        this_image_path = get_specific_image(this_index, this_B, this_T, this_S, Z, C, this_M)
                        if this_image_path is not None and os.path.exists(this_image_path):
                            sum_pic_in_this_S += 1
                            if this_M not in tiles_margin:
                                sum_silly_S += 1
                                print('<<<--- Block:', this_B, 'Time:', this_T, 'Scene:', this_S, 'Mosaic Tiles:',
                                      this_M, end='')
                                density_sum += get_AE_density(this_index, this_B, this_T, this_S, Z, C, this_M)
                                print(' --->>>')
                        else:
                            print('Block:', this_B, 'Time:', this_T, 'Scene:', this_S,
                                  'has ZEN auto export problems! missing:', this_M)
                    if sum_silly_S != 0:
                        print('Silly: sum of Scene:', sum_silly_S)
                        avg_density = density_sum / sum_silly_S
                        print('orig   each images avg:', avg_density)
                        saving_density_RT(main_path, 'AVG_Density.csv', this_index, this_T, this_S, avg_density)
                    if sum_pic_in_this_S == args_M:
                        stitching_result = square_stitching_save(main_path, this_index, this_B, this_T, this_S, Z, C,
                                                                 args_M, square_side, zoom, overlap)
                        zws_density = get_img_density(stitching_result[0])
                        znes_density = get_img_density(stitching_result[1])
                        print('zoom   whole stitching:', zws_density)
                        print('zoom no edge stitching:', znes_density)
                        saving_density_RT(main_path, 'Whole_Well_Density.csv', this_index, this_T, this_S, zws_density)
                        saving_density_RT(main_path, 'No_Edge_Density.csv', this_index, this_T, this_S, znes_density)
                    cardinal_mem.loc[this_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = [
                        this_B, this_T, this_S, Z, C, this_M]
                    cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
                elif silly == 0:  # silly==0: Normal: If images missing DO NOT ( Calculate avg_density & Stitching )! ;
                    sum_pic_in_this_S = 0
                    for this_M in range(1, args_M + 1):
                        this_image_path = get_specific_image(this_index, this_B, this_T, this_S, Z, C, this_M)
                        if this_image_path is not None and os.path.exists(this_image_path):
                            sum_pic_in_this_S += 1
                        else:
                            print('Block:', this_B, 'Time:', this_T, 'Scene:', this_S, ' missing:', this_M, '!!!')
                    if sum_pic_in_this_S == args_M:
                        saving_time(main_path, 'Calculation_Time.csv', this_index, this_T, this_S)
                        # calculate the avg density
                        density_sum = 0
                        for this_M in range(1, args_M + 1):
                            if this_M not in tiles_margin:
                                print('<<<--- Block:', this_B, 'Time:', this_T, 'Scene:', this_S, 'Mosaic Tiles:',
                                      this_M, end='')
                                density_sum += get_AE_density(this_index, this_B, this_T, this_S, Z, C, this_M)
                                print(' --->>>')
                        avg_density = density_sum / (args_M - len(tiles_margin))
                        # stitching and stitching
                        stitching_result = square_stitching_save(main_path, this_index, this_B, this_T, this_S, Z, C,
                                                                 args_M, square_side, zoom, overlap)
                        # whole picture calculate
                        # stitching_result[0] is the whole stitching (a well)
                        # stitching_result[1] is the 9 square
                        # if need some image analysis do at here
                        zws_density = get_img_density(stitching_result[0])
                        znes_density = get_img_density(stitching_result[1])
                        print('orig   each images avg:', avg_density)
                        print('zoom   whole stitching:', zws_density)
                        print('zoom no edge stitching:', znes_density)
                        # saving_scene_density_csv(main_path, this_index, this_T, this_S, avg_density)
                        saving_density_RT(main_path, 'AVG_Density.csv', this_index, this_T, this_S, avg_density)
                        saving_density_RT(main_path, 'Whole_Well_Density.csv', this_index, this_T, this_S, zws_density)
                        saving_density_RT(main_path, 'No_Edge_Density.csv', this_index, this_T, this_S, znes_density)
                        if Density_Alarm == 1:
                            if len(density_avg_base_list) < 10:
                                density_avg_base_list.append(znes_density)
                            else:
                                density_avg_base_list.pop(0)
                                density_avg_base_list.append(znes_density)
                                density_avg_base = np.mean(density_avg_base_list)
                            if density_avg_base >= 0.76:
                                # Convergence degree 75% Alarm!
                                os.system(
                                    "start python Make_a_Call.py --phoneNum {} --content {}".format('17710805067',
                                                                                                    '4F60597DFF0C6211662F004300440037FF0C0049005000536C4754085EA68FBE5230767E52064E4B4E0353414E94FF0C8BF751C6590763626DB2'))
                        # saving Cardinal.csv ...
                        cardinal_mem.loc[this_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = [
                            this_B, this_T, this_S, Z, C, this_M]
                        cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
                    elif this_S == this_end_loop_S:
                        print('The latest unfinished Scene!')
                    else:
                        print('Block:', this_B, 'Time:', this_T, 'Scene:', this_S, 'has ZEN auto export problems! :',
                              sum_pic_in_this_S, 'export!')
                        print('Discard this Scene!!!')
                        next
    cardinal_mem.loc[this_index, ['analysis_complete']] = 1
    cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
    return True


def m3_rt(this_index, args_B, args_T, args_S, args_Z, args_C, args_M):
    global main_path, tiles_margin, cardinal_mem, MAX_PYTHON_process_number, TIME_SLICE, image_file_type, square_side, zoom, overlap, silly, Density_Alarm, density_avg_base_list
    while True:
        time_start = time.time()

        if os.path.exists(os.path.join(this_index, 'ExperimentComplete.txt')):
            with open(os.path.join(this_index, 'ExperimentComplete.txt'), 'r') as fin:
                ec_str = fin.read()
            os.remove(os.path.join(this_index, 'ExperimentComplete.txt'))
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
            time.sleep(TIME_SLICE)
    return True


def main(args):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.width', None)

    print('Hello I am the Scheduling_Sequential.py program! Now I will sleep 5s!')
    time.sleep(5)

    global main_path, tiles_margin, cardinal_mem, MAX_PYTHON_process_number, TIME_SLICE, image_file_type, square_side, zoom, overlap, silly, Density_Alarm, density_avg_base_list
    main_path = args.main_path
    tiles_margin = scan_dish_margin(main_path)
    cardinal_mem = pd.read_csv(os.path.join(main_path, 'Cardinal.csv'), header=0, index_col=0)
    cardinal_mem = cardinal_mem.fillna(0)
    cardinal_mem = cardinal_mem.applymap(lambda x: int(x))
    # scene_density_mem = pd.read_csv(os.path.join(main_path, 'Scene_Density.csv'), header=0, index_col=0)
    MAX_PYTHON_process_number = args.max_process
    TIME_SLICE = args.time_slice
    image_file_type = ('.jpg', '.png', '.tif')
    square_side = 5
    zoom = [args.zoom, 0.3]
    # zoom can be a float or a list, the first number is the processing pic zoom size,the rest is only store
    overlap = args.overlap
    silly = args.silly
    Density_Alarm = 0
    density_avg_base_list = []

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
            # print('Mission Complete!!!')
            # return True
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
            # print('Mission Complete!!!')
            # return True
        else:
            print('Method 4 ERROR! Wrong using method 4!')
            time.sleep(30)
            exit('Method 4 ERROR! Wrong using method 4!')

    print('Mission Complete!!!')
    time.sleep(30)


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', type=str, nargs='?',
                        default=r'C:\Users\Kitty\Desktop\Cardinal-Test\CD11_examples', help='The Main Folder Path')
    parser.add_argument('--B', type=int, nargs='?', default=1, help='B=block')
    parser.add_argument('--T', type=int, nargs='?', default=-1, help='T=time')
    parser.add_argument('--S', type=int, nargs='?', default=96, help='S=scene')
    parser.add_argument('--Z', type=int, nargs='?', default=3, help='Z=z-direction')
    parser.add_argument('--C', type=int, nargs='?', default=1, help='C=channel')
    parser.add_argument('--M', type=int, nargs='?', default=25, help='M=mosaic tiles')
    parser.add_argument('--max_process', type=int, nargs='?', default=10, help='max_process')
    parser.add_argument('--time_slice', type=int, nargs='?', default=15, help='time_slice')
    parser.add_argument('--zoom', type=float, nargs='?', default=1, help='Stitching whole image resize zoom')
    parser.add_argument('--overlap', type=float, nargs='?', default=0.1, help='Stitching overlap')
    parser.add_argument('--silly', type=int, nargs='?', default=0, help='silly == 0 : normal!')
    # silly==0: Normal: If images missing DO NOT ( Calculate avg_density & Stitching )! ;
    # silly==1: Calculate avg_density ANYWAY! & If images missing DO NOT Stitching! ;
    # silly==2: Calculate avg_density ANYWAY! & DO NOT Stitching! ;
    # silly==3: Calculate avg_density ANYWAY! & Stitching ANYWAY! ;
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
