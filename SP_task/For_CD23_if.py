import sys

sys.path.append("..")
from Lib_Tiles import return_CD23_Tiles
from Lib_Function import stitching_CZI_IEed_AutoBestZ_bat, stitching_CZI_IEed_AutoBestZ_spS_bat


main_path = r'D:\PROCESSING\CD23'
path=r'D:\PROCESSING\CD23\2019-05-29\CD23_A_D5end_IF'
B = 1
C = 1
zoom = 1
overlap = 0.05
matrix_list = return_CD23_Tiles()
stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                 name_C=True)





