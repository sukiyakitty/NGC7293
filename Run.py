import os
import sys
import time
import argparse
import shutil
import numpy as np
import pandas as pd
from Lib_Function import stitching_CZI_IEed_AutoBestZ_bat, stitching_CZI_IEed_AutoBestZ_allC_bat, \
    stitching_CZI_IEed_AutoBestZ_spS_allfolder_bat, stitching_CZI, stitching_CZI_IEed_allZ_bat
from Lib_Features import research_stitched_image_elastic_bat, shading_correction, shading_correction_IF, \
    make_CD11_shading_correction_IF, make_CD44_shading_correction, make_CD44_copy_to_seg, make_CD11_copy_to_seg, \
    make_CD13_copy_to_seg, make_SC002_copy_to_seg, make_SC006_copy_to_seg, make_CD23_copy_to_seg, make_CD26_copy_to_seg, \
    make_CD33_copy_to_seg, extract_Fractal, make_CD58_copy_to_seg, make_CD61_copy_to_seg
from Lib_Tiles import return_CD11_Tiles, return_CD13_Tiles, return_96well_25_Tiles
from Lib_Sort import files_sort_CD11, files_sort_CD13, files_sort_univers


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


def change_labeled_pic_name(from_path, csv_file):
    if not os.path.exists(from_path):
        print('!ERROR! The from_path does not existed!')
        return False
    if not os.path.exists(csv_file):
        print('!ERROR! The csv_file does not existed!')
        return False

    df_csv = pd.read_csv(csv_file, header=0)
    df_csv = df_csv.fillna('error')

    path_list = os.listdir(from_path)

    for i in path_list:  # r'Label_1.png'
        old_img_file = os.path.join(from_path, i)

        label = int(i.split('.')[0].split('Label_')[1])
        S = df_csv.loc[label - 1, 'Using_S']

        if not S == 'error':
            new_img_file = os.path.join(from_path, S + '~label.png')
            os.rename(old_img_file, new_img_file)

    return True


def change_names(from_path):
    if not os.path.exists(from_path):
        print('!ERROR! The from_path does not existed!')
        return False

    path_list = os.listdir(from_path)

    for i in path_list:  # r'Label_1.png'
        old_img_file = os.path.join(from_path, i)
        # new_name = i.split('~')[0]+'~End.png'
        new_name = i.split('~')[0] + '~' + i.split('~')[-1]
        # new_name = i.split('~')[0]+'~Z'+i.split('~')[-1]
        new_img_file = os.path.join(from_path, new_name)
        os.rename(old_img_file, new_img_file)

    return True


def CD58_change_names_and_spli():
    from_path = r'T:\CD58'

    if not os.path.exists(from_path):
        print('!ERROR! The from_path does not existed!')
        return False

    folder_Day5 = 'hand_labeling_Day5'
    folder_End = 'hand_labeling_End'
    folder_IF = 'hand_labeling_IF'

    if not os.path.exists(os.path.join(r'T:\CD58A', folder_Day5)):
        os.makedirs(os.path.join(r'T:\CD58A', folder_Day5))
    if not os.path.exists(os.path.join(r'T:\CD58A', folder_End)):
        os.makedirs(os.path.join(r'T:\CD58A', folder_End))
    if not os.path.exists(os.path.join(r'T:\CD58A', folder_IF)):
        os.makedirs(os.path.join(r'T:\CD58A', folder_IF))

    if not os.path.exists(os.path.join(r'T:\CD58B', folder_Day5)):
        os.makedirs(os.path.join(r'T:\CD58B', folder_Day5))
    if not os.path.exists(os.path.join(r'T:\CD58B', folder_End)):
        os.makedirs(os.path.join(r'T:\CD58B', folder_End))
    if not os.path.exists(os.path.join(r'T:\CD58B', folder_IF)):
        os.makedirs(os.path.join(r'T:\CD58B', folder_IF))

    this_folder = os.path.join(from_path, folder_Day5)
    path_list = os.listdir(this_folder)
    for i in path_list:  # r'S1~CD58_d6_cpc_A~Z1.png'
        old_img_file = os.path.join(this_folder, i)
        # new_name = i.split('~')[0]+'~End.png'
        AB = i.split('~')[1].split('_cpc_')[-1]
        new_name = i.split('~')[0] + '~' + i.split('~')[-1]
        # new_name = i.split('~')[0]+'~Z'+i.split('~')[-1]
        if AB == 'A':
            new_img_file = os.path.join(r'T:\CD58A', folder_Day5, new_name)
        else:
            new_img_file = os.path.join(r'T:\CD58B', folder_Day5, new_name)
        # os.rename(old_img_file, new_img_file)
        shutil.copy(old_img_file, new_img_file)

    this_folder = os.path.join(from_path, folder_End)
    path_list = os.listdir(this_folder)
    for i in path_list:  # r'S1~CD58_d6_cpc_A~Z1.png'
        old_img_file = os.path.join(this_folder, i)
        # new_name = i.split('~')[0]+'~End.png'
        AB = i.split('~')[1].split('CD58')[-1].split('_')[0]
        new_name = i.split('~')[0] + '~End.png'
        # new_name = i.split('~')[0]+'~Z'+i.split('~')[-1]
        if AB == 'A':
            new_img_file = os.path.join(r'T:\CD58A', folder_End, new_name)
        else:
            new_img_file = os.path.join(r'T:\CD58B', folder_End, new_name)
        # os.rename(old_img_file, new_img_file)
        shutil.copy(old_img_file, new_img_file)

    this_folder = os.path.join(from_path, folder_IF)
    path_list = os.listdir(this_folder)
    for i in path_list:  # r'S1~CD58_d6_cpc_A~Z1.png'
        old_img_file = os.path.join(this_folder, i)
        # new_name = i.split('~')[0]+'~End.png'
        AB = i.split('~')[1].split('CD58')[-1].split('2_')[0]
        new_name = i.split('~')[0] + '~' + i.split('~')[-1]
        # new_name = i.split('~')[0]+'~Z'+i.split('~')[-1]
        if AB == 'A':
            new_img_file = os.path.join(r'T:\CD58A', folder_IF, new_name)
        else:
            new_img_file = os.path.join(r'T:\CD58B', folder_IF, new_name)
        # os.rename(old_img_file, new_img_file)
        shutil.copy(old_img_file, new_img_file)

    return True


if __name__ == '__main__':


    main_path = r'D:\CD61'
    extract_Fractal(main_path, times=15, sort_function=files_sort_univers, sleep_s=1, my_title='', usingPGC=True,
                    uingIPS=True)

    # main_path = r'D:\CD61'
    # analysis_function = make_CD61_copy_to_seg
    # zoom = 1
    # sort_function = None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)
    
    # main_path = r'D:\CD61'
    # B = 1
    # T = 1
    # all_S = 96
    # all_Z = 5
    # all_C = 3
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.15
    # sort_function = None
    # output = None
    # path = r'D:\CD61\2021-05-24\CD59_d6_cpc'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=True, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # path = r'D:\CD61\2021-05-31\CD61_d12_liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)
    # path = r'D:\CD61\2021-06-12\CD61_IF'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)

    # main_path = r'D:\CD58\Processing'
    # B = 1
    # T = 1
    # all_S = 48
    # all_Z = 7
    # all_C = 1
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.15
    # sort_function = None
    # output = None
    # path = r'D:\CD58\Processing\2021-04-04\CD58A_Result'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=True,
    #                             name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # path = r'D:\CD58\Processing\2021-04-05\CD58B_Result'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=True,
    #                             name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)

    # CD58_change_names_and_spli()
    # main_path = r'D:\CD58\Processing'
    # analysis_function = make_CD58_copy_to_seg
    # zoom = 1
    # sort_function = None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)

    # main_path = r'D:\CD58\Processing'
    # B = 1
    # T = 1
    # all_S = 48
    # all_Z = 5
    # all_C = 1
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.15
    # sort_function = None
    # output = None
    # path = r'D:\CD58\Processing\2021-03-30\CD58_d6_cpc_A'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=True,
    #                             name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # path = r'D:\CD58\Processing\2021-03-30\CD58_d6_cpc_B'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=True,
    #                             name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)

    # main(parseArguments(sys.argv[1:]))

    # change_labeled_pic_name(r'C:\Users\Kitty\Desktop\PixelLabelData', r'C:\Users\Kitty\Desktop\CD13_mask_2.csv')
    # main_path = r'D:\CD55\Processing'
    # extract_Fractal(main_path, times=15, sort_function=files_sort_univers, sleep_s=1, my_title='', usingPGC=True,
    #                 uingIPS=True)

    # main_path = r'T:\CD23'
    # analysis_function = make_CD23_copy_to_seg
    # zoom = 1
    # sort_function = None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)
    # change_names(r'T:\CD23\hand_labeling_IF')
    # from_path=r'T:\CD23\PixelLabelData'
    # csv_file=r'T:\CD23\CD23_mask.csv'
    # change_labeled_pic_name(from_path, csv_file)

    # change_names(r'T:\Image_Processing\hand_labeling\CD11\hand_labeling_Mask')
    # change_names(r'T:\Image_Processing\hand_labeling\CD13\hand_labeling_Mask')
    # change_names(r'T:\Image_Processing\hand_labeling\SC02\hand_labeling_Mask')

    # main_path = r'T:\scrambleIPS18_002'
    # analysis_function = make_SC002_copy_to_seg
    # zoom = 1
    # sort_function = None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)

    # main_path = r'T:\CD13'
    # analysis_function = make_CD13_copy_to_seg
    # zoom = 1
    # sort_function = None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)

    # main_path = r'T:\CD13'
    # B = 1
    # T = 44
    # all_S = 96
    # all_Z = 3
    # all_C = 3
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.05
    # sort_function = None
    # output = None
    # path = r'T:\CD13\III-1_CD13'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=T, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)

    # from_path=r'T:\scrambleIPS18_002\PixelLabelData'
    # csv_file=r'T:\scrambleIPS18_002\SC002_mask.csv'
    # change_labeled_pic_name(from_path, csv_file)

    # main_path = r'T:\CD33'
    # analysis_function = make_CD33_copy_to_seg
    # zoom = 1
    # sort_function = None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)
    #
    # main_path = r'T:\CD26'
    # analysis_function = make_CD26_copy_to_seg
    # zoom = 1
    # sort_function = None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)

    # main_path = r'T:\CD33'
    # B = 1
    # T = 1
    # all_S = 96
    # all_Z = 3
    # all_C = 3
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.05
    # sort_function = None
    # output = None
    # path = r'T:\CD33\CD33_STAGE2_144H_[D5end_S12]'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=True,
    #                             name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # path = r'T:\CD33\CD33_STAGE3_270H_[Beating]'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)
    # path = r'T:\CD33\CD33_STAGE3_280H_[Result_IF]'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)
    #
    # main_path = r'T:\CD26'
    # B = 1
    # T = 17
    # all_S = 96
    # all_Z = 4
    # all_C = 3
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.05
    # sort_function = None
    # output = None
    # path = r'T:\CD26\CD26_STAGEII_D5'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=True,
    #                             name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # path = r'T:\CD26\CD26_End'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)
    # path = r'T:\CD26\CD26_Result_IF'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)

    # from_path=r'T:\CD13\PixelLabelData'
    # csv_file=r'T:\CD13\CD13_mask.csv'
    # change_labeled_pic_name(from_path, csv_file)

    # main_path = r'T:\CD23'
    # analysis_function = make_CD23_copy_to_seg
    # zoom = 1
    # sort_function = None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)

    # main_path= r'T:\CD23'
    # B = 1
    # T = 1
    # all_S = 96
    # all_Z = 1
    # all_C = 3
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.05
    # sort_function = None
    # output = None
    #
    # path = r'T:\CD23\CD23_B_D5end'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=True,
    #                             name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    #
    # path = r'T:\CD23\CD23_B_D11_5slices'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)
    #
    # path = r'T:\CD23\CD23_B_Result'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)

    # main_path = r'G:\scrambleIPS18_002'
    # analysis_function = make_SC002_copy_to_seg
    # zoom=1
    # sort_function=None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)
    #
    # main_path = r'G:\scrambleIPS18_006'
    # analysis_function = make_SC006_copy_to_seg
    # zoom = 1
    # sort_function = None
    # research_stitched_image_elastic_bat(main_path, zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)

    # main_path = r'G:\scrambleIPS18_006'
    # B = 1
    # T = 1
    # all_S = 96
    # all_Z = 3
    # all_C = 3
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.05
    # sort_function = None
    # output = None
    #
    # path = r'G:\scrambleIPS18_006\2020-12-14\scrambleIPS18_D6_006'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=True,
    #                             name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    #
    # path = r'G:\scrambleIPS18_006\2020-12-22\scrambleIPS18_D13_liveCM_006'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)
    #
    # path = r'G:\scrambleIPS18_006\2021-01-03\scrambleIPS18_006_cTNT_3'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)

    # main_path = r'G:\scrambleIPS18_002'
    # B = 1
    # T = 1
    # all_S = 48
    # all_Z = 3
    # all_C = 3
    # C = 1
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.05
    # sort_function = None
    # output =None
    #
    # path = r'G:\scrambleIPS18_002\2020-11-30\scrambleIPS18_D6_002'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    #                       name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False, do_enhancement=False)
    #
    # path = r'G:\scrambleIPS18_002\2020-12-07\scrambleIPS18_D13_liveCM_002'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)
    #
    # path = r'G:\scrambleIPS18_002\2020-12-20\scrambleIPS18_D13_002_cTNT'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)

    # do_3 = ['2018-11-24~2018-11-24_Result_CD11~T1~C1', '2018-11-24~2018-11-24_Result_CD11~T1~C2',
    #       '2018-11-24~2018-11-24_Result_CD11~T1~C3']
    # do_4 = ['2018-11-13~CD11~Z1', '2018-11-13~CD11~Z2', '2018-11-13~CD11~Z3']

    # from_path = r'C:\Users\Zeiss User\Desktop\CD13\II-3_CD13'
    # path_list = os.listdir(from_path)
    # for i in path_list:
    #     img_file = os.path.join(from_path, i)
    #     # i.split('~')[1] # S1~2018-11-24_Result_CD11~T1~C1.png S3~CD11~Z1
    #     if i.split('~')[1] == '2018-11-24_Result_CD11':
    #         to_file = os.path.join(r'C:\Users\Zeiss User\Desktop\CD11\hand_labeling_Result', i)
    #         shutil.copy(img_file, to_file)
    #     if i.find('~CD11~Z') != -1:
    #         to_file = os.path.join(r'C:\Users\Zeiss User\Desktop\CD11\hand_labeling_Z123', i)
    #         shutil.copy(img_file, to_file)

    # output = r'I:\CD44\PROCESSING\Shading_corrected'
    # analysis_function = shading_correction

    # path = r'H:\CD44\PROCESSING\CD44_Stage-2_120H'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=24, S=1, output=output,
    #                                  do_SSSS=True, name_C=False)

    # path = r'T:\CD13\II-3_CD13'
    # for S in range(1, all_S + 1):
    #     for Z in range(1, all_Z + 1):
    #         stitching_CZI(main_path, path, B, T, S, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    #                       name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False, do_enhancement=False)

    # path = r'T:\CD13\result_parallel_CD13'
    # for S in range(1, all_S + 1):
    #     for Z in range(1, all_Z + 1):
    #         stitching_CZI(main_path, path, B, T, S, Z, C, matrix_list, zoom, overlap, output=None, do_SSSS=True,
    #                       name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False, do_enhancement=False)

    # path = r'C:\Users\Zeiss User\Desktop\CD11\2018-11-13_1200_II-3_CD11_T24'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=24, S=1, output=None,
    #                                  do_SSSS=True, name_C=False)

    # research_stitched_image_elastic_bat(output, zoom, analysis_function, sort_function, do_SSS=True, do_SSSS=True,
    #                                     do_parallel=False, process_number=12)

    # output = r'T:\Image_Processing\CD11\results_IF'
    # main_path = r'C:\Users\Zeiss User\Desktop\CD11\hand_labeling'

    # path = r'T:\Image_Processing\CD11\2018-11-24\2018-11-24_Result_CD11'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=output,
    #                                       do_SSSS=True)
    #
    # path = r'T:\CD13\result_parallel_CD13'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    #                                       do_SSSS=True)
