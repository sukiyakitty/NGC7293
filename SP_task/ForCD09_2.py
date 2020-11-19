import sys

sys.path.append("..")
from Lib_Tiles import return_CD09_ORCA_Tiles, return_CD09_506_Tiles
from Lib_Function import stitching_CZI_IEed_AutoBestZ_bat, stitching_CZI_IEed_AutoBestZ_spS_bat

main_path = r'C:\Users\Kitty\Desktop\CD09'
B = 1
C = 2
sp_S = 9
# sp_S = 1 to 24
# matrix_list =return_CD09_ORCA_Tiles()
# matrix_list =return_CD09_506_Tiles()
zoom = [1, 0.5, 0.25, 0.125]
overlap = 0.1

path = r'G:\CD09\Exported\2018-09-17~Result_CD09'
matrix_list = return_CD09_506_Tiles()
# main_path, path, B, sp_S, C, matrix_list, zoom, overlap, output=None,do_SSSS=True, name_C=False
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 9, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 10, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 11, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 12, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 13, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 14, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 15, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 16, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 17, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)

stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 18, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 19, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 20, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 21, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 22, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 23, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
stitching_CZI_IEed_AutoBestZ_spS_bat(main_path, path, B, 24, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                     name_C=True)
