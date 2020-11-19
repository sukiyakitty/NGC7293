import os
import sys
import time
import argparse
import pandas as pd
from Lib_Function import get_cpu_python_process, scan_dish_margin


# cardinal_mem = pd.DataFrame(
#     columns=['scheduling_method', 'experiment_complete', 'exp_B', 'exp_T', 'exp_S', 'exp_Z', 'exp_C', 'exp_M',
#              'analysis_complete', 'ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M'])


def sub_task_feature_extraction(path1, path2, this_B, this_T, this_S, this_Z, this_C, this_M):
    os.system(
        "start /b python Task_Extract_MFeatures.py --main_path {} --this_path {} --this_B {} --this_T {} --this_S {} --this_Z {} --this_C {} --this_M {}".format(
            path1, path2, this_B, this_T, this_S, this_Z, this_C, this_M))


def call_RT_Scheduling(args_path, args_B, args_T, args_S, args_Z, args_C, args_M, args_rt_process, args_max_process,
                       args_time_slice):
    os.system(
        "start python Scheduling_RealTime.py --main_path {} --B {} --T {} --S {} --Z {} --C {} --M {} --rt_process {} --max_process {} --time_slice {}".format(
            args_path, args_B, args_T, args_S, args_Z, args_C, args_M, args_rt_process, args_max_process,
            args_time_slice))


def row_scheduling(this_F_index, this_F_row, args_B, args_T, args_S, args_Z, args_C, args_M):
    global main_path, tiles_margin, cardinal_mem, ONCE_PYTHON_number
    execute_number = 0
    if this_F_row['ana_B'] == 0:
        this_start_loop_B = 1
    else:
        this_start_loop_B = this_F_row['ana_B']
    this_end_loop_B = this_F_row['exp_B']
    # B always 1 ! Different B, Different TSZCM, following do NOT Consider B:
    for this_B in range(this_start_loop_B, this_end_loop_B + 1):
        this_start_loop_M = this_F_row['ana_M']
        if this_F_row['ana_M'] == 0:
            this_start_loop_M = 1
        this_end_loop_M = args_M  # do not consider last unfinished well
        for this_M in range(this_start_loop_M, this_end_loop_M + 1):
            if this_M not in tiles_margin:
                if this_M == this_F_row['ana_M']:
                    this_start_loop_T = this_F_row['ana_T']
                    if this_F_row['ana_T'] == 0:
                        this_start_loop_T = 1
                else:
                    this_start_loop_T = 1
                if args_T == -1:
                    this_end_loop_T = this_F_row['exp_T']
                else:
                    if this_B == this_F_row['exp_B']:
                        this_end_loop_T = this_F_row['exp_T']
                    else:
                        this_end_loop_T = args_T
                for this_T in range(this_start_loop_T, this_end_loop_T + 1):
                    if this_M == this_F_row['ana_M'] and this_T == this_F_row['ana_T']:
                        this_start_loop_S = this_F_row['ana_S'] + 1
                    else:
                        this_start_loop_S = 1
                    if this_T == this_F_row['exp_T']:
                        if this_F_row['exp_M'] == args_M and this_F_row['exp_Z'] == args_Z and this_F_row[
                            'exp_C'] == args_C:
                            this_end_loop_S = this_F_row['exp_S']
                        else:
                            this_end_loop_S = this_F_row['exp_S'] - 1
                        if this_end_loop_S == 0:
                            pass
                    else:
                        this_end_loop_S = args_S
                    for this_S in range(this_start_loop_S, this_end_loop_S + 1):  # note ZEN bugs
                        execute_number += 1
                        print('<<<---', execute_number, '---', this_F_index, this_B, this_T, this_S, 2, 1, this_M,
                              end='')
                        sub_task_feature_extraction(main_path, this_F_index, this_B, this_T, this_S, 2, 1, this_M)
                        cardinal_mem.loc[this_F_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = [
                            this_B, this_T, this_S, 2, 1, this_M]
                        cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
                        print(' --->>>')
                        if execute_number == ONCE_PYTHON_number:
                            return False
    cardinal_mem.loc[this_F_index, ['analysis_complete']] = 1
    cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
    return True


def col_scheduling(this_F_index, this_F_row, args_B, args_T, args_S, args_Z, args_C, args_M):
    global main_path, tiles_margin, cardinal_mem, ONCE_PYTHON_number
    execute_number = 0
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
                this_start_loop_S = this_F_row['ana_S']
            else:
                this_start_loop_S = 1
            if this_T == this_F_row['exp_T']:
                this_end_loop_S = this_F_row['exp_S']
            else:
                this_end_loop_S = args_S
            for this_S in range(this_start_loop_S, this_end_loop_S + 1):
                if this_T == this_F_row['ana_T'] and this_S == this_F_row['ana_S']:
                    this_start_loop_M = this_F_row['ana_M'] + 1
                else:
                    this_start_loop_M = 1
                if this_T == this_F_row['exp_T'] and this_S == this_F_row['exp_S']:
                    this_end_loop_M = this_F_row['exp_M']
                else:
                    this_end_loop_M = args_M
                for this_M in range(this_start_loop_M, this_end_loop_M + 1):
                    if this_M not in tiles_margin:
                        execute_number += 1
                        print('<<<---', execute_number, '---', this_F_index, this_B, this_T, this_S, 2, 1, this_M,
                              end='')
                        sub_task_feature_extraction(main_path, this_F_index, this_B, this_T, this_S, 2, 1, this_M)
                        cardinal_mem.loc[this_F_index, ['ana_B', 'ana_T', 'ana_S', 'ana_Z', 'ana_C', 'ana_M']] = [
                            this_B, this_T, this_S, 2, 1, this_M]
                        cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
                        print(' --->>>')
                        if execute_number == ONCE_PYTHON_number:
                            return False
    cardinal_mem.loc[this_F_index, ['analysis_complete']] = 1
    cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
    return True


def main(args):
    global main_path, tiles_margin, cardinal_mem, ONCE_PYTHON_number, MAX_PYTHON_process_number
    ONCE_PYTHON_number = args.once_process
    MAX_PYTHON_process_number = args.max_process
    RT_Scheduling = False
    main_path = args.main_path
    tiles_margin = scan_dish_margin(main_path)
    cardinal_mem = pd.read_csv(os.path.join(main_path, 'Cardinal.csv'), header=0, index_col=0)
    cardinal_mem = cardinal_mem.applymap(lambda x: int(x))
    for this_RT_index, this_RT_row in cardinal_mem.loc[
        (cardinal_mem['experiment_complete'] == 0) & (cardinal_mem['scheduling_method'] == 2) & (
                cardinal_mem['analysis_complete'] == 0)].iterrows():
        call_RT_Scheduling(main_path, args.B, args.T, args.S, args.Z, args.C, args.M, args.rt_process, args.max_process,
                           args.time_slice)
        RT_Scheduling = True
    rt_count = 0
    while True:
        time_start = time.time()
        # for row_scheduling
        for this_F_index, this_F_row in cardinal_mem.loc[
            (cardinal_mem['experiment_complete'] == 1) & (cardinal_mem['scheduling_method'] == 1) & (
                    cardinal_mem['analysis_complete'] == 0)].iterrows():
            row_scheduling(this_F_index, this_F_row, args.B, args.T, args.S, args.Z, args.C, args.M)
            temp_p_number = get_cpu_python_process()
            print('$$$---Python Number: ', temp_p_number, '---$$$')
            if temp_p_number > MAX_PYTHON_process_number:
                print('!@#$%^&*(NOW sleep!)*&^%$#@!')
                time.sleep(2)
                while True:
                    if get_cpu_python_process() >= MAX_PYTHON_process_number:
                        time.sleep(2)
                    else:
                        break
        # for col_scheduling
        for this_F_index, this_F_row in cardinal_mem.loc[
            (cardinal_mem['experiment_complete'] == 1) & (cardinal_mem['scheduling_method'] == 2) & (
                    cardinal_mem['analysis_complete'] == 0)].iterrows():
            col_scheduling(this_F_index, this_F_row, args.B, args.T, args.S, args.Z, args.C, args.M)
            temp_p_number = get_cpu_python_process()
            print('$$$---Python Number: ', temp_p_number, '---$$$')
            if temp_p_number > MAX_PYTHON_process_number:
                print('!@#$%^&*(NOW sleep!)*&^%$#@!')
                time.sleep(2)
                while True:
                    if get_cpu_python_process() >= MAX_PYTHON_process_number:
                        time.sleep(2)
                    else:
                        break
        # # for RT_scheduling
        # time_duration = time.time() - time_start
        # if RT_Scheduling:
        #     if cardinal_mem['analysis_complete'].sum() == cardinal_mem['analysis_complete'].size - 1:
        #         if time_duration < 15:
        #             time.sleep(15 - time_duration)
        #     else:
        #         rt_count += 1
        #         if rt_count == 3:
        #             rt_count = 0
        #         else:
        #             next
        #     if os.path.exists(os.path.join(main_path, 'Cardinal_RTemp.csv')):
        #         cardinal_RTemp = pd.read_csv(os.path.join(main_path, 'Cardinal_RTemp.csv'), header=0,
        #                                      index_col=0)
        #         cardinal_mem.loc[this_RT_index] = cardinal_RTemp.loc[this_RT_index]
        #         cardinal_mem.to_csv(path_or_buf=os.path.join(main_path, 'Cardinal.csv'))
        #         print('collection complete!')
        #         if cardinal_RTemp.loc[this_RT_index, 'experiment_complete'] == 1 and cardinal_RTemp.loc[
        #             this_RT_index, 'analysis_complete'] == 1:
        #             RT_Scheduling = False
        if cardinal_mem['analysis_complete'].sum() == cardinal_mem['experiment_complete'].sum():
            break
    print('Mission Complete!!!')


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', type=str, nargs='?', default='D:\\CD11_small', help='The Main Folder Path')
    parser.add_argument('--B', type=int, nargs='?', default=1, help='B=block')
    parser.add_argument('--T', type=int, nargs='?', default=-1, help='T=time')
    parser.add_argument('--S', type=int, nargs='?', default=72, help='S=scene')
    parser.add_argument('--Z', type=int, nargs='?', default=3, help='Z=z-direction')
    parser.add_argument('--C', type=int, nargs='?', default=1, help='C=channel')
    parser.add_argument('--M', type=int, nargs='?', default=25, help='M=mosaic tiles')
    parser.add_argument('--rt_process', type=int, nargs='?', default=6, help='M=mosaic tiles')
    parser.add_argument('--once_process', type=int, nargs='?', default=3, help='M=mosaic tiles')
    parser.add_argument('--max_process', type=int, nargs='?', default=10, help='M=mosaic tiles')
    parser.add_argument('--time_slice', type=int, nargs='?', default=15, help='M=mosaic tiles')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
