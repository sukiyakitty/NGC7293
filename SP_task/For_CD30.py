import os
import numpy as np
import pandas as pd
import cv2
from Lib_Movies import any_to_image, make_texted_img


def func_image_x(input,output):
    # input = r'E:\CD30\PROCESSING\All_images\SSSS_100%'
    # output = r'E:\CD30\PROCESSING\Compare_images_SSSS'
    # count = None

    well_all = [i for i in range(1, 96 + 1)]
    CHIR = [4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 12,
            12, 12, 12, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12,
            8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4]
    time_A = [36] * 48 + [60] * 48
    time_B = [12] * 24 + [24] * 24 + [48] * 48
    result_A = [0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.5, 0.5, 0.65, 0.65, 0.8, 0.9, 0.3, 0.3, 0.3, 0.3, 0, 0, 0,
                0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.5, 0.5, 0.6, 0.6, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3, 0, 0, 0,
                0, 0, 0, 0, 0, 0.85, 0.85, 0.8, 0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0.85, 0.5, 0.85, 0.85, 0, 0, 0, 0, 0, 0,
                0, 0, 0.9, 0.9, 0.85, 0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0.85, 0.85, 0.85, 0.9, 0, 0, 0, 0]
    result_B = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0,
                0, 0, 0, 0, 0.85, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.85, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0,
                0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]

    dir_list = os.listdir(input)
    count = len(dir_list)
    for i in dir_list:
        i_list = i.split('~')
        i_S_number = int(i_list[0].split('S')[1])
        if i_S_number in well_all:

            time = int(i_list[2].split('_')[2].split('H')[0])
            chir = CHIR[i_S_number - 1]
            if (i_list[2].find('CD30(A)') == 0):
                chir_hour = time_A[i_S_number - 1]
                result = result_A[i_S_number - 1]
            elif (i_list[2].find('CD30(B)') == 0):
                chir_hour = time_B[i_S_number - 1]
                result = result_B[i_S_number - 1]

            # [time, chir, chir_hour, result]
            puton_str = str(time) + 'H CHIR' + str(chir) + ' Result:' + '%.0f%%' % (result * 100)
            # img = any_to_image(os.path.join(input, i))
            img = cv2.imread(os.path.join(input, i), 1)
            make_texted_img(img, puton_str, result)
            out_folder = os.path.join(output, str(time) + 'H')
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            out_name = os.path.join(out_folder, i)
            cv2.imwrite(out_name, img)
            count = count - 1
            print('>>> ' + str(count) + '  ' + i)

    return True


def func_0():
    main_path = r'C:\Users\Kitty\Documents\Desktop\CD30\PROCESSING'
    input_csv = r'MyPGC_Features.csv'
    output_csv = r'Classed_MyPGC_Features.csv'

    well_all = [i for i in range(1, 96 + 1)]

    # well_chir_4 = [i for i in range(1, 4 + 1)] + [i for i in range(21, 28 + 1)] + [i for i in range(45, 52 + 1)] + \
    #               [i for i in range(69, 76 + 1)] + [i for i in range(93, 96 + 1)]
    # well_chir_8 = [5, 6, 7, 8, 17, 18, 19, 20, 29, 30, 31, 32, 41, 42, 43, 44, 53, 54, 55, 56, 65, 66, 67, 68, 77, 78,
    #                79, 80, 89, 90, 91, 92]
    # well_chir_12 = [i for i in range(9, 16 + 1)] + [i for i in range(33, 40 + 1)] + [i for i in range(57, 64 + 1)] + \
    #                [i for i in range(81, 88 + 1)]
    # well_B_12h = [i for i in range(1, 24 + 1)]
    # well_B_24h = [i for i in range(25, 48 + 1)]
    # well_B_48h = [i for i in range(49, 96 + 1)]
    # well_A_36h = [i for i in range(1, 48 + 1)]
    # well_A_60h = [i for i in range(49, 96 + 1)]

    chir = [4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 12,
            12, 12, 12, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12,
            8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4]
    time_A = [36] * 48 + [60] * 48
    time_B = [12] * 24 + [24] * 24 + [48] * 48

    result_A = [0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.5, 0.5, 0.65, 0.65, 0.8, 0.9, 0.3, 0.3, 0.3, 0.3, 0, 0, 0,
                0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.5, 0.5, 0.6, 0.6, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3, 0, 0, 0,
                0, 0, 0, 0, 0, 0.85, 0.85, 0.8, 0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0.85, 0.5, 0.85, 0.85, 0, 0, 0, 0, 0, 0,
                0, 0, 0.9, 0.9, 0.85, 0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0.85, 0.85, 0.85, 0.9, 0, 0, 0, 0]
    result_B = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0,
                0, 0, 0, 0, 0.85, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.85, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0,
                0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]

    # well_scr = []
    # well_good = []
    # well_bad = []
    # well_mid = []

    input_csv_path = os.path.join(main_path, input_csv)
    output_csv_path = os.path.join(main_path, output_csv)
    all_data = pd.read_csv(input_csv_path, header=0, index_col=0)
    result_data = pd.DataFrame()

    for i in all_data.index:  # 'S1~2019-07-15~CD30(A)_STAGE0_-1H_[IPS18]~T1'
        i_list = i.split('~')  # ['S1', '2019-07-15', 'CD30(A)_STAGE0_-1H_[IPS18]', 'T1']
        i_S_number = int(i_list[0].split('S')[1])
        if i_S_number in well_all:

            a_class = int(i_list[2].split('_')[2].split('H')[0])
            a_chir = chir[i_S_number - 1]
            if (i_list[2].find('CD30(A)') == 0):
                a_chir_hour = time_A[i_S_number - 1]
                a_result = result_A[i_S_number - 1]
            elif (i_list[2].find('CD30(B)') == 0):
                a_chir_hour = time_B[i_S_number - 1]
                a_result = result_B[i_S_number - 1]

            this_Series = all_data.loc[i]
            this_Series = this_Series.append(
                pd.Series([a_class, a_chir, a_chir_hour, a_result],
                          index=['a_class', 'a_chir', 'a_chir_hour', 'a_result'], name=i))
            result_data = result_data.append(this_Series)

            # if (a_chir == 8 and (a_chir_hour == 60 or (a_chir_hour < 60 and a_class < 60))) or a_class == -1:
            #     this_Series = all_data.loc[i]
            #     this_Series = this_Series.append(
            #         pd.Series([a_class, a_chir, a_chir_hour, a_result],
            #                   index=['a_class', 'a_chir', 'a_chir_hour', 'a_result'], name=i))
            #     result_data = result_data.append(this_Series)

    result_data.to_csv(path_or_buf=os.path.join(main_path, output_csv_path))


def func_1():
    main_path = r'C:\Users\Kitty\Documents\Desktop\CD30\PROCESSING'
    input_csv = r'CD30_predict.csv'
    output_csv = r'CD30_predict_wells.csv'

    input_csv_path = os.path.join(main_path, input_csv)
    output_csv_path = os.path.join(main_path, output_csv)
    all_data = pd.read_csv(input_csv_path, header=0, index_col=0)
    temp_data = pd.DataFrame()

    for i in all_data.index:  # 'S1~2019-07-15~CD30(A)_STAGE0_-1H_[IPS18]~T1'
        i_list = i.split('~')  # ['S1', '2019-07-15', 'CD30(A)_STAGE0_-1H_[IPS18]', 'T1']
        new_index_name = i_list[2].split('_')[0] + '~' + 'S%02d' % int(i_list[0].split('S')[1])  # '%04d' % int(key)
        this_Series = all_data.loc[i]
        # this_Series.name = new_index_name
        this_Series = this_Series.append(
            pd.Series([new_index_name], index=['new_index_name'], name=i))
        temp_data = temp_data.append(this_Series)

    result_data = temp_data.groupby('new_index_name')['Speed'].sum()

    result_data.to_csv(path_or_buf=os.path.join(main_path, output_csv_path))

    return True


def func_2():
    main_path = r'C:\Users\Kitty\Documents\Desktop\CD30\PROCESSING'
    input_csv = r'CD30_predict.csv'
    output_csv = r'CD30_predict_wells_before_60h.csv'

    input_csv_path = os.path.join(main_path, input_csv)
    output_csv_path = os.path.join(main_path, output_csv)
    all_data = pd.read_csv(input_csv_path, header=0, index_col=0)
    temp_data = pd.DataFrame()

    for i in all_data.index:  # 'S1~2019-07-15~CD30(A)_STAGE0_-1H_[IPS18]~T1'
        i_list = i.split('~')  # ['S1', '2019-07-15', 'CD30(A)_STAGE0_-1H_[IPS18]', 'T1']

        if int(i_list[2].split('_')[2].split('H')[0]) <= 60:
            new_index_name = i_list[2].split('_')[0] + '~' + 'S%02d' % int(i_list[0].split('S')[1])  # '%04d' % int(key)
            this_Series = all_data.loc[i]
            # this_Series.name = new_index_name
            this_Series = this_Series.append(
                pd.Series([new_index_name], index=['new_index_name'], name=i))
            temp_data = temp_data.append(this_Series)

    result_data = temp_data.groupby('new_index_name')['Speed'].sum()

    result_data.to_csv(path_or_buf=os.path.join(main_path, output_csv_path))

    return True


if __name__ == '__main__':
    # func_2()

    # input = r'E:\CD30\PROCESSING\All_images\SSSS_100%'
    # output = r'E:\CD30\PROCESSING\Compare_images_SSSS'
    # func_image_x(input, output)
    input = r'E:\CD30\PROCESSING\Enhanced_img\All_images\SSS_100%'
    output = r'E:\CD30\PROCESSING\Enhanced_img\Compare_images_SSS'
    func_image_x(input, output)
    input = r'E:\CD30\PROCESSING\MyPGC_img\All_images\SSS_100%'
    output = r'E:\CD30\PROCESSING\MyPGC_img\Compare_images_SSS'
    func_image_x(input, output)

