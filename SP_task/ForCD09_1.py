import sys
sys.path.append("..")
from Lib_Tiles import return_CD09_ORCA_Tiles, return_CD09_506_Tiles
from Lib_Function import stitching_CZI_IEed_AutoBestZ_bat, stitching_CZI_IEed_AutoBestZ_spS_bat

main_path = r'C:\Users\Kitty\Desktop\CD09'
B = 1
C = 1
# sp_S = 1 to 24
# matrix_list =return_CD09_ORCA_Tiles()
# matrix_list =return_CD09_506_Tiles()
zoom = [1, 0.5, 0.25, 0.125]
overlap = 0.1

# path = r'G:\CD09\Exported\2018-09-04~I-3_CD09'
# matrix_list = return_CD09_ORCA_Tiles()
# stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, output=None, do_SSSS=False)
#
# path = r'G:\CD09\Exported\2018-09-05~I-4_CD09'
# matrix_list = return_CD09_ORCA_Tiles()
# stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, output=None, do_SSSS=False)

path = r'G:\CD09\Exported\2018-09-05~I-5_CD09'
matrix_list = return_CD09_506_Tiles()
stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, output=None, do_SSSS=False,
                                 name_C=True)

# path = r'G:\CD09\Exported\2018-09-05~I-6_CD09'
# matrix_list = return_CD09_ORCA_Tiles()
# stitching_CZI_IEed_AutoBestZ_bat(main_path, path, B, C, matrix_list, zoom, overlap, output=None, do_SSSS=False)
