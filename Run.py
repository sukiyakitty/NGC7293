import os
import sys
import time
import argparse


def main(args):
    print('!Notice! This is NOT the main function running!')

    os.system(
        "start python Scheduling_x.py --main_path {} --B {} --T {} --S {} --Z {} --C {} --M {} --time_slice {} --zoom {} --overlap {} --analysis {}".format(
            r'C:\Users\Kitty\Desktop\CD22', 1, -1, 96, 3, 1, 25, 30, 1, 0.05, 7))
    # main_path=r'J:\PROCESSING\CD11'
    # if movie_exp_bat(main_path, 0.3, mov_width=2000, mov_height=2000):
    #     print('Done!!!')

    # if move_merge_image_folder(r'J:\PROCESSING\CD13\test',r'J:\PROCESSING\CD13',1):
    #     print('Done!!!')

    # if extract_sp_image(args.main_path, 0):  # 1 means the experiment has biological repetition
    #     print('Done!!!')

    # index_path = []
    # index_path.append(r'D:\PROCESSING\CD16\2019-01-13\IPS_CD16')
    # index_path.append(r'D:\PROCESSING\CD16\2019-01-14\I-1_CD16')
    # image_resize(args.main_path, index_path, 1, 0.3)
    #
    # os.system("start python Make_a_Call.py --phoneNum {}".format('17710805067'))
    # os.system(
    #     "start python Make_a_Call.py --phoneNum {} --content {}".format('17710805067', '4F60597DFF0C6D4B8BD551855BB9'))
    # os.system(
    #     "start python Make_a_Call.py --phoneNum {} --content {}".format('17710805067',
    #                                                                     '4F60597DFF0C6211662F004300440037FF0C0049005000536C4754085EA68FBE5230767E52064E4B4E0353414E94FF0C8BF751C6590763626DB2'))


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', type=str, nargs='?', default=r'D:\PROCESSING\CD16', help='The Main Folder Path')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
