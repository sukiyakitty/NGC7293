import os
import sys
import time
import argparse
import shutil
import numpy as np
import pandas as pd
from Lib_Function import stitching_CZI_IEed_AutoBestZ_bat, stitching_CZI_IEed_AutoBestZ_allC_bat, \
    stitching_CZI_IEed_AutoBestZ_spS_allfolder_bat, stitching_CZI, stitching_CZI_IEed_allZ_bat,add_prefix
from Lib_Features import research_image_bat, research_stitched_image_elastic_bat, shading_correction, \
    shading_correction_IF, features_write_all, feature_write_entropy, merge_all_well_features, feature_write_3, \
    transform_matrix_features_to_diff_vector, research_image_bat_continue, make_copy_to_destination, \
    make_CD11_shading_correction_IF, make_CD44_shading_correction, make_CD44_copy_to_seg, make_CD11_copy_to_seg, \
    make_CD13_copy_to_seg, make_SC002_copy_to_seg, make_SC006_copy_to_seg, make_CD23_copy_to_seg, make_CD26_copy_to_seg, \
    make_CD33_copy_to_seg, extract_Fractal, make_CD58_copy_to_seg, make_CD61_copy_to_seg
from Lib_Tiles import return_CD11_Tiles, return_CD13_Tiles, return_96well_25_Tiles, return_384well_9_Tiles
from Lib_Sort import files_sort_CD11, files_sort_CD13, files_sort_CD26, files_sort_univers, files_sort_CD46
from Lib_Manifold import do_manifold, do_manifold_for_multi_batch, return_lda_ref_DF
from Lib_Visualization import draw_mainfold_elastic_inOneFolder_bat, CD13_All_wells, first_phase_first10hours, \
    first10hours_relative_CHIR_proposal_by_time, draw_mainfold_bat, relative_CHIR_proposal_by_time, \
    all_wells_colored_by_IF_only_SP_time, relative_CHIR_proposal_by_time_and_test, \
    all_wells_colored_by_IF, CD13_All_Success_wells_IFhuman_GE05, CD13_All_Failure_wells_IFhuman_L01, \
    CD13_Diffrent_Stages_improved


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


def CD13_conbi():
    in1 = r'E:\Image_Processing\CD13\Entropys_fisrt10hours'
    in2 = r'E:\Image_Processing\CD13\Features_fisrt10hours'
    out = r'E:\Image_Processing\CD13\Features3'
    if os.path.exists(out):
        shutil.rmtree(out)
        pass
    else:
        os.makedirs(out)  # make the output folder

    in1_list = os.listdir(in1)
    for i in in1_list:
        in1_file = os.path.join(in1, i)
        in2_file = os.path.join(in2, i)
        finall_file = os.path.join(out, i)
        in1_file_PD = pd.read_csv(in1_file, header=0, index_col=0)
        in2_file_PD = pd.read_csv(in2_file, header=0, index_col=0)

        finall_DF = in1_file_PD
        finall_DF['density'] = in2_file_PD['density']
        finall_DF['fractal'] = in2_file_PD['fractal']
        finall_DF.to_csv(path_or_buf=finall_file)

    return True


if __name__ == '__main__':

    analysis_function = make_copy_to_destination
    zoom = 1
    sort_function = None
    research_stitched_image_elastic_bat(r'L:\CD63\Processing', zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True)
    research_stitched_image_elastic_bat(r'L:\CD64\Processing', zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True)
    research_stitched_image_elastic_bat(r'L:\CD65\Processing', zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True)
    research_stitched_image_elastic_bat(r'L:\CD26', zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True)
    research_stitched_image_elastic_bat(r'L:\CD44', zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True)
    research_stitched_image_elastic_bat(r'L:\CD33', zoom, analysis_function, sort_function, do_SSS=False, do_SSSS=True)

    add_prefix(r'L:\CD63\Processing\A', r'CD63')
    add_prefix(r'L:\CD63\Processing\B', r'CD63')
    add_prefix(r'L:\CD64\Processing\A', r'CD64')
    add_prefix(r'L:\CD64\Processing\B', r'CD64')
    add_prefix(r'L:\CD65\Processing\A', r'CD65')
    add_prefix(r'L:\CD65\Processing\B', r'CD65')
    add_prefix(r'L:\CD26\A', r'CD26')
    add_prefix(r'L:\CD26\B', r'CD26')
    add_prefix(r'L:\CD44\A', r'CD44')
    add_prefix(r'L:\CD44\B', r'CD44')
    add_prefix(r'L:\CD33\A', r'CD33')
    add_prefix(r'L:\CD33\B', r'CD33')

    # B = 1
    # all_C = 3
    # matrix_list = return_96well_25_Tiles()
    # zoom = 1
    # overlap = 0.15
    #
    # main_path = r'L:\CD63\Processing'
    # path = r'L:\CD63\Processing\2021-07-01\CD63_IF'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap)
    #
    # main_path = r'L:\CD64\Processing'
    # path = r'L:\CD64\Processing\2021-07-02\CD64A_IF'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap)
    #
    # main_path = r'L:\CD65\Processing'
    # path = r'L:\CD65\Processing\2021-07-23\CD65_IF'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap)
    #
    # overlap = 0.05
    # main_path = r'L:\CD26'
    # path = r'L:\CD26\CD26_Result_IF'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap)
    #
    # overlap = 0.05
    # main_path = r'L:\CD44'
    # path = r'L:\CD44\CD44_Result-IF'
    # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap)


    # 384 bat -------------------------------------------------------
    # B = 1
    # T = 1
    # all_S = 384
    # all_Z = 3
    # all_C = 1
    # C = 1
    # matrix_list = return_384well_9_Tiles()
    # zoom = 1
    # overlap = 0.1
    #
    # main_path = r'D:\SM-384\SM-384well-01-d11-liveCM'
    # path = r'D:\SM-384\SM-384well-01-d11-liveCM\2021-08-12\SM-384well-01-d11-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-02-d15-liveCM'
    # path = r'D:\SM-384\SM-384well-02-d15-liveCM\2021-08-15\SM-384well-02-d15-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-04-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-04-d6-liveCPC\2021-08-15\SM-384well-04-d6-liveCPC'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-04-d13-liveCM'
    # path = r'D:\SM-384\SM-384well-04-d13-liveCM\2021-08-22\SM-384well-04-d13-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-05-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-05-d6-liveCPC\2021-08-14\SM-384well-05-d6-liveCPC'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-05-d12-liveCM'
    # path = r'D:\SM-384\SM-384well-05-d12-liveCM\2021-08-20\SM-384well-05-d12-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-06-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-06-d6-liveCPC\2021-08-20\SM-384well-06-d6-liveCPC'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-06-d15-liveCM'
    # path = r'D:\SM-384\SM-384well-06-d15-liveCM\2021-08-29\SM-384well-06-d15-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-07-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-07-d6-liveCPC\2021-08-21\SM-384well-07-d6-liveCPC'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-07-d12-liveCM'
    # path = r'D:\SM-384\SM-384well-07-d12-liveCM\2021-08-26\SM-384well-07-d12-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-08-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-08-d6-liveCPC\2021-08-21\SM-384well-08-d6-liveCPC'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-08-d13-liveCM'
    # path = r'D:\SM-384\SM-384well-08-d13-liveCM\2021-08-27\SM-384well-08-d13-liveCM-02'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-09-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-09-d6-liveCPC\2021-08-23\SM-384well-09-d6-liveCPC'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-09-d13-liveCM'
    # path = r'D:\SM-384\SM-384well-09-d13-liveCM\2021-08-31\SM-384well-09-d13-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-10-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-10-d6-liveCPC\2021-08-23\SM-384well-10-d6-liveCPC-02'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-10-d15-liveCM'
    # path = r'D:\SM-384\SM-384well-10-d15-liveCM\2021-09-02\SM-384well-10-d15-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-11-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-11-d6-liveCPC\2021-08-24\SM-384well-11-d6-liveCPC'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-11-d15-liveCM'
    # path = r'D:\SM-384\SM-384well-11-d15-liveCM\2021-09-04\SM-384well-11-d15-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)
    # main_path = r'D:\SM-384\SM-384well-11-d17-liveCM'
    # path = r'D:\SM-384\SM-384well-11-d17-liveCM\2021-09-06\SM-384well-11-d17-liveCM'
    # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    #                             do_SSSS=False, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    #                             do_enhancement=False)

    # 384 bat -------------------------------------------------------
    # B = 1
    # T = 1
    # all_S = 384
    # all_Z = 3
    # all_C = 1
    # C = 1
    # matrix_list = return_384well_9_Tiles()
    # zoom = 1
    # overlap = 0.1
    #
    # main_path = r'D:\SM-384\SM-384well-02-d15-liveCM'
    # path = r'D:\SM-384\SM-384well-02-d15-liveCM\2021-08-15\SM-384well-02-d15-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-04-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-04-d6-liveCPC\2021-08-15\SM-384well-04-d6-liveCPC'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-04-d13-liveCM'
    # path = r'D:\SM-384\SM-384well-04-d13-liveCM\2021-08-22\SM-384well-04-d13-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-05-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-05-d6-liveCPC\2021-08-14\SM-384well-05-d6-liveCPC'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-05-d12-liveCM'
    # path = r'D:\SM-384\SM-384well-05-d12-liveCM\2021-08-20\SM-384well-05-d12-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-06-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-06-d6-liveCPC\2021-08-20\SM-384well-06-d6-liveCPC'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-06-d15-liveCM'
    # path = r'D:\SM-384\SM-384well-06-d15-liveCM\2021-08-29\SM-384well-06-d15-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-07-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-07-d6-liveCPC\2021-08-21\SM-384well-07-d6-liveCPC'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-07-d12-liveCM'
    # path = r'D:\SM-384\SM-384well-07-d12-liveCM\2021-08-26\SM-384well-07-d12-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-08-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-08-d6-liveCPC\2021-08-21\SM-384well-08-d6-liveCPC'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-08-d13-liveCM'
    # path = r'D:\SM-384\SM-384well-08-d13-liveCM\2021-08-27\SM-384well-08-d13-liveCM-02'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-09-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-09-d6-liveCPC\2021-08-23\SM-384well-09-d6-liveCPC'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-09-d13-liveCM'
    # path = r'D:\SM-384\SM-384well-09-d13-liveCM\2021-08-31\SM-384well-09-d13-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-10-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-10-d6-liveCPC\2021-08-23\SM-384well-10-d6-liveCPC-02'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-10-d15-liveCM'
    # path = r'D:\SM-384\SM-384well-10-d15-liveCM\2021-09-02\SM-384well-10-d15-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-11-d6-liveCPC'
    # path = r'D:\SM-384\SM-384well-11-d6-liveCPC\2021-08-24\SM-384well-11-d6-liveCPC'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-11-d15-liveCM'
    # path = r'D:\SM-384\SM-384well-11-d15-liveCM\2021-09-04\SM-384well-11-d15-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # main_path = r'D:\SM-384\SM-384well-11-d17-liveCM'
    # path = r'D:\SM-384\SM-384well-11-d17-liveCM\2021-09-06\SM-384well-11-d17-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # ------------------------------------------------------------------------------------------------------------

    # 384-wells stitching  --------------------------------------------------
    # B = 1
    # T = 1
    # all_S = 384
    # all_Z = 3
    # all_C = 1
    # C = 1
    # matrix_list = return_384well_9_Tiles()
    # zoom = 1
    # overlap = 0.1
    # # sort_function = None
    # # output = None
    # # path = r''
    # # stitching_CZI_IEed_allZ_bat(main_path, path, B, T, all_S, all_Z, C, matrix_list, zoom, overlap, output=None,
    # #                             do_SSSS=True, name_B=False, name_T=False, name_S=False, name_Z=True, name_C=False,
    # #                             do_enhancement=False)
    # main_path = r'D:\SM-384well-01-d11-liveCM'
    # path = r'D:\SM-384well-01-d11-liveCM\2021-08-12\SM-384well-01-d11-liveCM'
    # stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, T=1, S=1, output=None,
    #                                  do_SSSS=False, name_C=False)
    # # path = r''
    # # stitching_CZI_IEed_AutoBestZ_allC_bat(main_path, path, B, all_C, matrix_list, zoom, overlap, output=None,
    # #                                       do_SSSS=True)
    # ------------------------------------------------------------------------------------------------------------

    # a whloe anlysis pip-line of CD68 with new multi-batch code --------------------------------------------------
    # main_path = r'X:\CD68'
    # Slist_folder = r'X:\CD68\SSSS_100%'
    # CD68_fisrt10 = ['2021-10-18~CD68_IPS-1~T1', '2021-10-18~CD68_STAGE-1_1H~T1', '2021-10-18~CD68_STAGE-1_1H~T2',
    #                 '2021-10-18~CD68_STAGE-1_1H~T3', '2021-10-18~CD68_STAGE-1_1H~T4', '2021-10-18~CD68_STAGE-1_1H~T5',
    #                 '2021-10-18~CD68_STAGE-1_1H~T6', '2021-10-18~CD68_STAGE-1_1H~T7', '2021-10-18~CD68_STAGE-1_1H~T8',
    #                 '2021-10-18~CD68_STAGE-1_1H~T9', '2021-10-18~CD68_STAGE-1_1H~T10']
    # research_image_bat(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD68_fisrt10,
    #                    sort_function=files_sort_univers)
    # # research_image_bat_continue(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD65_fisrt10,
    # #                             sort_function=files_sort_univers, S=??)
    # features_path = r'Feature3'
    # transform_matrix_features_to_diff_vector(main_path, features_path,
    #                                          output_csv='diff_vector_Features3_fisrt10hours.csv')

    # ------------------------------------------------------------------------------------------------------------

    # a whloe anlysis pip-line of CD43 with new multi-batch code --------------------------------------------------
    # main_path = r'E:\Image_Processing\CD43'
    # Slist_folder = r'E:\Image_Processing\CD43\SSSS_100%'
    # CD43_fisrt10 = ['2020-06-29~CD43_IPS-3~T2', '2020-06-29~CD43_Stage-1_1H~T1', '2020-06-29~CD43_Stage-1_1H~T2',
    #                 '2020-06-29~CD43_Stage-1_1H~T3', '2020-06-29~CD43_Stage-1_1H~T4', '2020-06-29~CD43_Stage-1_1H~T5',
    #                 '2020-06-29~CD43_Stage-1_1H~T6', '2020-06-29~CD43_Stage-1_1H~T7', '2020-06-29~CD43_Stage-1_1H~T8',
    #                 '2020-06-29~CD43_Stage-1_1H~T9', '2020-06-29~CD43_Stage-1_1H~T10']
    # research_image_bat(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD43_fisrt10,
    #                    sort_function=files_sort_univers)
    # # research_image_bat_continue(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD65_fisrt10,
    # #                             sort_function=files_sort_univers, S=??)
    # features_path = r'Feature3'
    # transform_matrix_features_to_diff_vector(main_path, features_path,
    #                                          output_csv='diff_vector_Features3_fisrt10hours.csv')

    # main_path = r'F:\CD65'
    # name_list = [r'CD65']
    # features_file_list = [r'diff_vector_Features3_fisrt10hours.csv']
    # features_cols = range(0, 30)
    # exp_file_list = []
    # exp_file_list += [os.path.join(main_path, r'Experiment_Plan.csv')]
    # output_path = os.path.join(main_path, r'MainFold_fisrt10hours_30cols')
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list))
    #
    # main_path = r'F:\CD65'
    # name_list = [r'CD65']
    # mainfold_path = os.path.join(main_path, r'MainFold_fisrt10hours_30cols')
    # exp_file_list = []
    # exp_file_list += [os.path.join(main_path, r'Experiment_Plan.csv')]
    # function_list = [relative_CHIR_proposal_by_time, all_wells_colored_by_IF]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # ------------------------------------------------------------------------------------------------------------

    # CD13_conbi()

    # a whloe anlysis pip-line of CD67 with new multi-batch code --------------------------------------------------
    # main_path = r'X:\CD67'
    # Slist_folder = r'X:\CD67\SSSS_100%'
    # CD67_fisrt10 = ['2021-09-29~CD67_IPS-1~T16', '2021-09-30~CD67_Stage-1_1H~T1', '2021-09-30~CD67_Stage-1_1H~T2',
    #                 '2021-09-30~CD67_Stage-1_1H~T3', '2021-09-30~CD67_Stage-1_1H~T4','2021-09-30~CD67_Stage-1_1H~T5', '2021-09-30~CD67_Stage-1_1H~T6',
    #                 '2021-09-30~CD67_Stage-1_1H~T7', '2021-09-30~CD67_Stage-1_1H~T8', '2021-09-30~CD67_Stage-1_1H~T9',
    #                 '2021-09-30~CD67_Stage-1_1H~T10']
    # research_image_bat(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD67_fisrt10,
    #                    sort_function=files_sort_univers)
    # # research_image_bat_continue(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD65_fisrt10,
    # #                             sort_function=files_sort_univers, S=??)
    # features_path = r'Feature3'
    # transform_matrix_features_to_diff_vector(main_path, features_path,
    #                                          output_csv='diff_vector_Features3_fisrt10hours.csv')

    # main_path = r'F:\CD65'
    # name_list = [r'CD65']
    # features_file_list = [r'diff_vector_Features3_fisrt10hours.csv']
    # features_cols = range(0, 30)
    # exp_file_list = []
    # exp_file_list += [os.path.join(main_path, r'Experiment_Plan.csv')]
    # output_path = os.path.join(main_path, r'MainFold_fisrt10hours_30cols')
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list))
    #
    # main_path = r'F:\CD65'
    # name_list = [r'CD65']
    # mainfold_path = os.path.join(main_path, r'MainFold_fisrt10hours_30cols')
    # exp_file_list = []
    # exp_file_list += [os.path.join(main_path, r'Experiment_Plan.csv')]
    # function_list = [relative_CHIR_proposal_by_time, all_wells_colored_by_IF]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # ------------------------------------------------------------------------------------------------------------

    # do manifold and test for multi batch  and TEST !!! ----------------------------------------------------------
    # name_list = ['CD13']
    # features_file_list = []
    # features_file_list += [r'E:\Image_Processing\CD13\diff_vector_Features3_fisrt10hours.csv']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # output_path = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26'
    # test_name_list = ['CD26']
    # test_features_file_list = []
    # test_features_file_list += [r'E:\Image_Processing\CD26\diff_vector_Features3_fisrt10hours.csv']
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list),
    #                             test_name_list=test_name_list, test_features_file_list=test_features_file_list)

    # name_list = ['CD13']
    # test_name_list = ['CD26']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # test_exp_file_list = []
    # test_exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    #
    # manifold_file = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\LDA_24H_LDA.csv'
    # TEST_manifold_file = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\TEST_LDA_24H_LDA.csv'
    # output_png = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\LDA_24H_LDA.png'
    # relative_CHIR_proposal_by_time_and_test(manifold_file, TEST_manifold_file, output_png, name_list, exp_file_list,
    #                                         test_name_list, test_exp_file_list, figsize=(12.80, 10.24), x_min=None,
    #                                         x_max=None,
    #                                         y_min=None, y_max=None)
    #
    # manifold_file = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\LDA_chir_diff_LDA.csv'
    # TEST_manifold_file = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\TEST_LDA_chir_diff_LDA.csv'
    # output_png = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\LDA_chir_diff_LDA.png'
    # relative_CHIR_proposal_by_time_and_test(manifold_file, TEST_manifold_file, output_png, name_list, exp_file_list,
    #                                         test_name_list, test_exp_file_list, figsize=(12.80, 10.24), x_min=None,
    #                                         x_max=None,
    #                                         y_min=None, y_max=None)
    #
    # manifold_file = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\LDA_36H_LDA.csv'
    # TEST_manifold_file = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\TEST_LDA_36H_LDA.csv'
    # output_png = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\LDA_36H_LDA.png'
    # relative_CHIR_proposal_by_time_and_test(manifold_file, TEST_manifold_file, output_png, name_list, exp_file_list,
    #                                         test_name_list, test_exp_file_list, figsize=(12.80, 10.24), x_min=None,
    #                                         x_max=None,
    #                                         y_min=None, y_max=None)
    #
    #
    # manifold_file = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\LDA_48H_LDA.csv'
    # TEST_manifold_file = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\TEST_LDA_48H_LDA.csv'
    # output_png = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_TEST_CD26\LDA_48H_LDA.png'
    # relative_CHIR_proposal_by_time_and_test(manifold_file, TEST_manifold_file, output_png, name_list, exp_file_list,
    #                                         test_name_list, test_exp_file_list, figsize=(12.80, 10.24), x_min=None,
    #                                         x_max=None,
    #                                         y_min=None, y_max=None)

    # -------------------------------------------------------------------------------------------------------------

    # do manifold and test for multi batch  and TEST !!! ----------------------------------------------------------
    # name_list = ['CD13', 'CD26']
    # features_file_list = []
    # features_file_list += [r'E:\Image_Processing\CD13\diff_vector_Features3_fisrt10hours.csv']
    # features_file_list += [r'E:\Image_Processing\CD26\diff_vector_Features3_fisrt10hours.csv']
    # # features_file_list += [r'E:\Image_Processing\CD44\diff_vector_Features3_fisrt10hours.csv']
    # # features_file_list += [r'E:\Image_Processing\CD46\diff_vector_Features3_fisrt10hours.csv']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # # exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # # exp_file_list += [os.path.join(r'E:\Image_Processing\CD46', r'Experiment_Plan.csv')]
    # output_path = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46'
    # test_name_list=['CD44', 'CD46']
    # test_features_file_list = []
    # test_features_file_list += [r'E:\Image_Processing\CD44\diff_vector_Features3_fisrt10hours.csv']
    # test_features_file_list += [r'E:\Image_Processing\CD46\diff_vector_Features3_fisrt10hours.csv']
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list),
    #                             test_name_list=test_name_list, test_features_file_list=test_features_file_list)

    # manifold_file=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\LDA_24H_LDA.csv'
    # TEST_manifold_file=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\TEST_LDA_24H_LDA.csv'
    # output_png=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\LDA_24H_LDA.png'
    # name_list = ['CD13', 'CD26']
    # test_name_list = ['CD44', 'CD46']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # test_exp_file_list = []
    # test_exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # test_exp_file_list += [os.path.join(r'E:\Image_Processing\CD46', r'Experiment_Plan.csv')]
    # relative_CHIR_proposal_by_time_and_test(manifold_file, TEST_manifold_file, output_png, name_list, exp_file_list,
    #                                 test_name_list, test_exp_file_list, figsize=(12.80, 10.24), x_min=None, x_max=None,
    #                                 y_min=None, y_max=None)
    #
    # manifold_file=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\LDA_36H_LDA.csv'
    # TEST_manifold_file=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\TEST_LDA_36H_LDA.csv'
    # output_png=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\LDA_36H_LDA.png'
    # name_list = ['CD13', 'CD26']
    # test_name_list = ['CD44', 'CD46']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # test_exp_file_list = []
    # test_exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # test_exp_file_list += [os.path.join(r'E:\Image_Processing\CD46', r'Experiment_Plan.csv')]
    # relative_CHIR_proposal_by_time_and_test(manifold_file, TEST_manifold_file, output_png, name_list, exp_file_list,
    #                                 test_name_list, test_exp_file_list, figsize=(12.80, 10.24), x_min=None, x_max=None,
    #                                 y_min=None, y_max=None)
    #
    # manifold_file=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\LDA_48H_LDA.csv'
    # TEST_manifold_file=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\TEST_LDA_48H_LDA.csv'
    # output_png=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\LDA_48H_LDA.png'
    # name_list = ['CD13', 'CD26']
    # test_name_list = ['CD44', 'CD46']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # test_exp_file_list = []
    # test_exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # test_exp_file_list += [os.path.join(r'E:\Image_Processing\CD46', r'Experiment_Plan.csv')]
    # relative_CHIR_proposal_by_time_and_test(manifold_file, TEST_manifold_file, output_png, name_list, exp_file_list,
    #                                 test_name_list, test_exp_file_list, figsize=(12.80, 10.24), x_min=None, x_max=None,
    #                                 y_min=None, y_max=None)
    #
    # manifold_file=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\LDA_chir_diff_LDA.csv'
    # TEST_manifold_file=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\TEST_LDA_chir_diff_LDA.csv'
    # output_png=r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_TEST_CD44_CD46\LDA_chir_diff_LDA.png'
    # name_list = ['CD13', 'CD26']
    # test_name_list = ['CD44', 'CD46']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # test_exp_file_list = []
    # test_exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # test_exp_file_list += [os.path.join(r'E:\Image_Processing\CD46', r'Experiment_Plan.csv')]
    # relative_CHIR_proposal_by_time_and_test(manifold_file, TEST_manifold_file, output_png, name_list, exp_file_list,
    #                                 test_name_list, test_exp_file_list, figsize=(12.80, 10.24), x_min=None, x_max=None,
    #                                 y_min=None, y_max=None)
    # -------------------------------------------------------------------------------------------------------------

    # do manifold for multi batch 'CD13', 'CD26', 'CD44','CD46', 'CD65' -------------------------------------------
    # name_list = ['CD13', 'CD26', 'CD44', 'CD46', 'CD65']
    # features_file_list = [r'E:\Image_Processing\CD13\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD26\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD44\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD46\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD65\diff_vector_Features3_fisrt10hours.csv']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD46', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD65', r'Experiment_Plan.csv')]
    # output_path = r'D:\Green\Sub_Projects\ML_assists_hiPSC-CM\Multi_Batch_CD13_CD26_CD44_CD46_CD65'
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list))
    #
    #
    # mainfold_path = r'D:\Green\Sub_Projects\ML_assists_hiPSC-CM\Multi_Batch_CD13_CD26_CD44_CD46_CD65'
    # name_list = ['CD13', 'CD26', 'CD44', 'CD46', 'CD65']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD46', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD65', r'Experiment_Plan.csv')]
    # function_list = [all_wells_colored_by_IF, relative_CHIR_proposal_by_time, all_wells_colored_by_IF_only_SP_time]
    # # function_list = [all_wells_colored_by_IF, relative_CHIR_proposal_by_time]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # -------------------------------------------------------------------------------------------------------------

    # a whloe anlysis pip-line of CD65 with new multi-batch code --------------------------------------------------
    # main_path = r'F:\CD65'
    # Slist_folder = r'F:\CD65\SSSS_100%'
    # CD65_fisrt10 = ['2021-07-10~CD65_Stage-1_1H~T1', '2021-07-10~CD65_Stage-1_1H~T2', '2021-07-10~CD65_Stage-1_1H~T3',
    #                 '2021-07-10~CD65_Stage-1_1H~T4', '2021-07-10~CD65_Stage-1_1H~T5', '2021-07-10~CD65_Stage-1_1H~T6',
    #                 '2021-07-10~CD65_Stage-1_1H~T7', '2021-07-10~CD65_Stage-1_1H~T8', '2021-07-10~CD65_Stage-1_1H~T9',
    #                 '2021-07-10~CD65_Stage-1_1H~T10', '2021-07-10~CD65_Stage-1_1H~T11']
    # research_image_bat(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD65_fisrt10,
    #                    sort_function=files_sort_univers)
    # # research_image_bat_continue(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD65_fisrt10,
    # #                             sort_function=files_sort_univers, S=??)
    # features_path = r'Feature3'
    # transform_matrix_features_to_diff_vector(main_path, features_path,
    #                                          output_csv='diff_vector_Features3_fisrt10hours.csv')

    # main_path = r'E:\Image_Processing\CD65'
    # name_list = [r'CD65']
    # features_file_list = [os.path.join(main_path, r'diff_vector_Features3_fisrt10hours.csv')]
    # features_cols = range(0, 30)
    # exp_file_list = []
    # exp_file_list += [os.path.join(main_path, r'Experiment_Plan.csv')]
    # output_path = os.path.join(main_path, r'MainFold_fisrt10hours_30cols')
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list))
    #
    # main_path = r'E:\Image_Processing\CD65'
    # name_list = [r'CD65']
    # mainfold_path = os.path.join(main_path, r'MainFold_fisrt10hours_30cols')
    # exp_file_list = []
    # exp_file_list += [os.path.join(main_path, r'Experiment_Plan.csv')]
    # function_list = [relative_CHIR_proposal_by_time, all_wells_colored_by_IF]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # ------------------------------------------------------------------------------------------------------------

    # do manifold for multi batch ---------------------------------------------------------------------------------
    # name_list = ['CD13', 'CD26', 'CD44', 'CD46']
    # features_file_list = [r'E:\Image_Processing\CD13\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD26\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD44\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD46\diff_vector_Features3_fisrt10hours.csv']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD46', r'Experiment_Plan.csv')]
    # output_path = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_CD44_CD46'
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, normalize=True, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list),
    #                             test_name_list=name_list, test_features_file_list=features_file_list)
    #
    # mainfold_path = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_CD44_CD46'
    # name_list = ['CD13', 'CD26', 'CD44', 'CD46']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD46', r'Experiment_Plan.csv')]
    # function_list = [all_wells_colored_by_IF, relative_CHIR_proposal_by_time, all_wells_colored_by_IF_only_SP_time]
    # function_list = [all_wells_colored_by_IF, relative_CHIR_proposal_by_time]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # -------------------------------------------------------------------------------------------------------------

    # a whloe anlysis pip-line of CD46 with new multi-batch code --------------------------------------------------
    # main_path = r'E:\Image_Processing\CD46'
    # Slist_folder = r'E:\Image_Processing\CD46\SSSS_100%'
    # CD46_fisrt10 = ['CD46~IPS-1~T22', 'CD46~1H~T1', 'CD46~1H~T2',
    #                 'CD46~1H~T3', 'CD46~1H~T4', 'CD46~1H~T5',
    #                 'CD46~1H~T6', 'CD46~1H~T7', 'CD46~1H~T8',
    #                 'CD46~1H~T9', 'CD46~1H~T10']
    #
    # # research_image_bat(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD46_fisrt10,
    # #                    sort_function=files_sort_CD46)
    # research_image_bat_continue(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD46_fisrt10,
    #                             sort_function=files_sort_CD46, S=30)
    #
    # features_path = r'Feature3'
    # transform_matrix_features_to_diff_vector(main_path, features_path,
    #                                          output_csv='diff_vector_Features3_fisrt10hours.csv')
    #
    # features_csv = r'diff_vector_Features3_fisrt10hours.csv'
    # features_cols = range(0, 30)
    # output_folder = r'MainFold_fisrt10hours_30cols'
    # do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)
    #
    # mainfold_path = os.path.join(main_path, r'MainFold_fisrt10hours_30cols')
    # exp_file = os.path.join(main_path, r'Experiment_Plan.csv')
    # function_list = [relative_CHIR_proposal_by_time]
    # draw_mainfold_bat(mainfold_path, ['CD46'], [exp_file], function_list)
    # ------------------------------------------------------------------------------------------------------------

    # a whloe anlysis pip-line of CD43 with new multi-batch code --------------------------------------------------
    # main_path = r'C:\Users\Kitty\Desktop\CD43'
    # Slist_folder = r'C:\Users\Kitty\Desktop\CD43\SSSS_100%'
    # CD43_fisrt10 = ['2020-06-29~CD43_IPS-3~T2', '2020-06-29~CD43_Stage-1_1H~T1', '2020-06-29~CD43_Stage-1_1H~T2',
    #                 '2020-06-29~CD43_Stage-1_1H~T3', '2020-06-29~CD43_Stage-1_1H~T4', '2020-06-29~CD43_Stage-1_1H~T5',
    #                 '2020-06-29~CD43_Stage-1_1H~T6', '2020-06-29~CD43_Stage-1_1H~T7', '2020-06-29~CD43_Stage-1_1H~T8',
    #                 '2020-06-29~CD43_Stage-1_1H~T9', '2020-06-29~CD43_Stage-1_1H~T10']
    #
    # # research_image_bat(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD43_fisrt10,
    # #                    sort_function=files_sort_univers)
    # research_image_bat_continue(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD43_fisrt10,
    #                    sort_function=files_sort_univers, S=84)
    #
    # features_path = r'Feature3'
    # transform_matrix_features_to_diff_vector(main_path, features_path,
    #                                          output_csv='diff_vector_Features3_fisrt10hours.csv')
    #
    # features_csv = r'diff_vector_Features3_fisrt10hours.csv'
    # features_cols = range(0, 30)
    # output_folder = r'MainFold_fisrt10hours_30cols'
    # do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)
    # ------------------------------------------------------------------------------------------------------------

    # do manifold for multi batch ---------------------------------------------------------------------------------
    # name_list = [r'CD13', r'CD26', r'CD44']
    # features_file_list = [r'E:\Image_Processing\CD13\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD26\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD44\diff_vector_Features3_fisrt10hours.csv']
    # output_path = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_CD44'
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list))
    #
    # mainfold_path = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26_CD44'
    # name_list = [r'CD13', r'CD26', r'CD44']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD44', r'Experiment_Plan.csv')]
    # # function_list = [all_wells_colored_by_IF, relative_CHIR_proposal_by_time, all_wells_colored_by_IF_only_SP_time]
    # function_list = [all_wells_colored_by_IF, relative_CHIR_proposal_by_time]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # -------------------------------------------------------------------------------------------------------------

    # do manifold for multi batch ---------------------------------------------------------------------------------
    # name_list = [r'CD13', r'CD26']
    # features_file_list = [r'E:\Image_Processing\CD13\diff_vector_Features3_fisrt10hours.csv',
    #                       r'E:\Image_Processing\CD26\diff_vector_Features3_fisrt10hours.csv']
    # output_path = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26'
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list))
    #
    # mainfold_path = r'C:\Users\Kitty\Desktop\Multi_Batch_CD13_CD26'
    # name_list = [r'CD13', r'CD26']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # function_list = [all_wells_colored_by_IF, relative_CHIR_proposal_by_time]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # -------------------------------------------------------------------------------------------------------------

    # do manifold for one batch ---------------------------------------------------------------------------------
    # name_list = [r'CD13']
    # features_file_list = [r'E:\Image_Processing\CD13\diff_vector_Features3_fisrt10hours.csv']
    # output_path = r'C:\Users\Kitty\Desktop\One_Batch_CD13'
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # do_manifold_for_multi_batch(name_list, features_file_list, output_path, features_cols=None, n_neighbors=10,
    #                             n_components=3, lda_ref_DF=return_lda_ref_DF(name_list, exp_file_list))
    #
    # mainfold_path = r'C:\Users\Kitty\Desktop\One_Batch_CD13'
    # name_list = [r'CD13']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # function_list = [all_wells_colored_by_IF, relative_CHIR_proposal_by_time]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # -------------------------------------------------------------------------------------------------------------

    # a whloe anlysis pip-line of CD44 with new multi-batch code --------------------------------------------------
    # main_path = r'E:\Image_Processing\CD44'
    # Slist_folder = r'E:\Image_Processing\CD44\SSSS_100%'
    # CD44_fisrt10 = ['2020-07-10~CD44_IPS-1~T24', '2020-07-11~CD44_Stage-1_1H~T1', '2020-07-11~CD44_Stage-1_1H~T2',
    #                 '2020-07-11~CD44_Stage-1_1H~T3', '2020-07-11~CD44_Stage-1_1H~T4', '2020-07-11~CD44_Stage-1_1H~T5',
    #                 '2020-07-11~CD44_Stage-1_1H~T6', '2020-07-11~CD44_Stage-1_1H~T7', '2020-07-11~CD44_Stage-1_1H~T8',
    #                 '2020-07-11~CD44_Stage-1_1H~T9', '2020-07-11~CD44_Stage-1_1H~T10']
    #
    # # research_image_bat(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD44_fisrt10,
    # #                    sort_function=files_sort_univers)
    # research_image_bat_continue(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD44_fisrt10,
    #                    sort_function=files_sort_univers, S=51)
    #
    # features_path = r'Feature3'
    # transform_matrix_features_to_diff_vector(main_path, features_path,
    #                                          output_csv='diff_vector_Features3_fisrt10hours.csv')
    #
    # features_csv = r'diff_vector_Features3_fisrt10hours.csv'
    # features_cols = range(0, 30)
    # output_folder = r'MainFold_fisrt10hours_30cols'
    # do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)
    #
    # mainfold_path = os.path.join(main_path, r'MainFold_fisrt10hours_30cols')
    # exp_file = os.path.join(main_path, r'Experiment_Plan.csv')
    # function_list = [relative_CHIR_proposal_by_time]
    # draw_mainfold_bat(mainfold_path, ['CD44'], [exp_file], function_list)
    # ------------------------------------------------------------------------------------------------------------

    # Multi batch analysis pip-line ------------------------------------------------------------------------------
    # test
    # main_path = r'C:\Users\Kitty\Desktop\CD13'
    # mainfold_path = os.path.join(main_path, r'MainFold_fisrt10hours_30cols')
    # name_list = r'CD13'
    # exp_file_list = os.path.join(main_path, r'Experiment_Plan.csv')
    # function_list = [relative_CHIR_proposal_by_time]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # test
    # mainfold_path = r'C:\C137\Sub_Projects\ML_assists_hiPSC-CM\Multi_Batch_CD13_CD26'
    # name_list = [r'CD13', r'CD26']
    # exp_file_list = []
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD13', r'Experiment_Plan.csv')]
    # exp_file_list += [os.path.join(r'E:\Image_Processing\CD26', r'Experiment_Plan.csv')]
    # function_list = [all_wells_colored_by_IF, relative_CHIR_proposal_by_time]
    # draw_mainfold_bat(mainfold_path, name_list, exp_file_list, function_list)
    # ------------------------------------------------------------------------------------------------------------

    # a whloe anlysis pip-line of CD26 ----------------------------------------------------------------------------
    # main_path = r'E:\Image_Processing\CD26'
    # Slist_folder = r'E:\Image_Processing\CD26\SSSS_100%'
    # CD26_fisrt10 = ['2019-06-14~CD26_IPS(H9)~T1', '2019-06-14~CD26_STAGEI_0H~T1', '2019-06-14~CD26_STAGEI_0H~T2',
    #                 '2019-06-14~CD26_STAGEI_0H~T3', '2019-06-14~CD26_STAGEI_0H~T4', '2019-06-14~CD26_STAGEI_0H~T5',
    #                 '2019-06-14~CD26_STAGEI_0H~T6', '2019-06-14~CD26_STAGEI_0H~T7', '2019-06-14~CD26_STAGEI_0H~T8',
    #                 '2019-06-14~CD26_STAGEI_0H~T9', '2019-06-14~CD26_STAGEI_0H~T10']
    #
    # research_image_bat(main_path, Slist_folder, analysis_function=feature_write_3, name_filter=CD26_fisrt10,
    #                    sort_function=files_sort_CD26)
    #
    # features_path = r'Feature3'
    # # merge_all_well_features(main_path, features_path, output_name='All_Features_fisrt10hours.csv')
    # transform_matrix_features_to_diff_vector(main_path, features_path, output_csv='diff_vector_Features3_fisrt10hours.csv')
    #
    # features_csv = r'diff_vector_Features3_fisrt10hours.csv'
    # features_cols = range(0, 30)
    # output_folder = r'MainFold_fisrt10hours_30cols'
    # do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)
    #
    # mainfold_path = r'MainFold_fisrt10hours_30cols'
    # exp_file = r'Experiment_Plan.csv'
    #
    # function_list = [first10hours_relative_CHIR_proposal_by_time]
    # draw_mainfold_elastic_inOneFolder_bat(main_path, mainfold_path, exp_file, function_list)
    # ------------------------------------------------------------------------------------------------------------

    # a whloe anlysis pip-line of CD13 ---------------------------------------------------------------------------
    # main_path = r'E:\Image_Processing\CD13'
    # Slist_folder = r'E:\Image_Processing\CD13\SSSS_100%'
    # cd13_fisrt10 = ['2018-11-30~IPS-3_CD13~T18', '2018-12-01~I-1_CD13~T1', '2018-12-01~I-1_CD13~T2',
    #                 '2018-12-01~I-1_CD13~T3',
    #                 '2018-12-01~I-1_CD13~T4', '2018-12-01~I-1_CD13~T5', '2018-12-01~I-1_CD13~T6',
    #                 '2018-12-01~I-1_CD13~T7',
    #                 '2018-12-01~I-1_CD13~T8', '2018-12-01~I-1_CD13~T9', '2018-12-01~I-1_CD13~T10']
    #
    # research_image_bat(main_path, Slist_folder, analysis_function=feature_write_entropy, name_filter=cd13_fisrt10,
    #                    sort_function=files_sort_CD13)
    #
    # features_path = r'Entropys_fisrt10hours'
    # # merge_all_well_features(main_path, features_path, output_name='All_Features_fisrt10hours.csv')
    # transform_matrix_features_to_diff_vector(main_path, features_path, output_csv='diff_vector_Entropys.csv')

    # features_csv = r'diff_vector_Features_fisrt10hours.csv'
    # features_cols = range(0,30)
    # output_folder = r'MainFold_fisrt10hours_30cols'
    # do_manifold(main_path, features_csv, features_cols, output_folder, n_neighbors=10, n_components=3)

    # mainfold_path = r'MainFold_fisrt10hours_30cols'
    # exp_file = r'Experiment_Plan.csv'
    # function_list = [CD13_All_Success_wells_IFhuman_GE05, CD13_All_Failure_wells_IFhuman_L01]
    # function_list = [CD13_Diffrent_Stages]
    # function_list = [first_phase_first10hours_relative_CHIR_sort_by_time]
    # draw_mainfold_elastic_inOneFolder_bat(main_path, mainfold_path, exp_file, function_list)
    # ------------------------------------------------------------------------------------------------------------

    # main_path = r'D:\CD61'
    # extract_Fractal(main_path, times=15, sort_function=files_sort_univers, sleep_s=1, my_title='', usingPGC=True,
    #                 uingIPS=True)
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
