import os
# import numpy as np
import pandas as pd


def func():
    pass


if __name__ == '__main__':
    main_path = r'C:\Users\Kitty\Desktop\CD13'
    input_csv = r'Pyramid_All_FEATURES.csv'
    output_csv = r'All_FEATURES_well_good_Pyramid.csv'
    well_all = [i for i in range(1, 96 + 1)]
    well_scr = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    well_good = [13, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 52, 55, 57, 58,
                 59]
    well_bad = [1, 2, 10, 11, 15, 34, 48, 49, 60, 61, 62, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 80, 83, 84, 86,
                87, 88, 89, 90, 91, 92, 93, 94, 95, 96]
    well_mid = [3, 4, 5, 6, 7, 8, 9, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 50, 51, 53, 54, 56, 63, 64, 76, 77,
                78, 79, 81, 82, 85]
    input_csv_path = os.path.join(main_path, input_csv)
    output_csv_path = os.path.join(main_path, output_csv)
    all_data = pd.read_csv(input_csv_path, header=0, index_col=0)
    result_data = pd.DataFrame()

    for i in all_data.index:  # 'S1~2018-11-28~IPS_CD13~T1'
        # print(i)
        i_list = i.split('~')  # ['S96', '2018-12-10', 'III-1_CD13', 'T44']
        if int(i_list[0].split('S')[1]) in well_good:
            # print(type(all_data.loc[i]))
            if (i_list[2].find('IPS-3') == 0) and (3 <= int(i_list[3].split('T')[1]) <= 5):  # IPS
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([-20], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-1') == 0) and (2 <= int(i_list[3].split('T')[1]) <= 4):  # Day0 add CHIR 1~19
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([3], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-1') == 0) and (16 <= int(i_list[3].split('T')[1]) <= 18):  # Day0 end add CHIR 1~19
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([22], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-1') == 0) and (21 <= int(i_list[3].split('T')[1]) <= 23):  # 24h
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([27], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-1') == 0) and (25 <= int(i_list[3].split('T')[1]) <= 27):  # 30h
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([33], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-1') == 0) and (28 <= int(i_list[3].split('T')[1]) <= 30):  # 36h
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([39], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-1') == 0) and (31 <= int(i_list[3].split('T')[1]) <= 31):  # 42h
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([45], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-2') == 0) and (2 <= int(i_list[3].split('T')[1]) <= 4):  # 48h
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([51], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-2') == 0) and (6 <= int(i_list[3].split('T')[1]) <= 8):  # 54h
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([57], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-2') == 0) and (15 <= int(i_list[3].split('T')[1]) <= 17):  # Day2 CHIR rest 5~18
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([70], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-2') == 0) and (20 <= int(i_list[3].split('T')[1]) < 22):  # Day3 19~30
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([74], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('I-2') == 0) and (26 <= int(i_list[3].split('T')[1]) <= 28):  # Day3 19~30
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([98], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('II-2') == 0) and (7 <= int(i_list[3].split('T')[1]) <= 9):  # Day4 1~10
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([118], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('II-2') == 0) and (15 <= int(i_list[3].split('T')[1]) <= 17):  # Day5 11~18
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([140], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

            if (i_list[2].find('III-1') == 0) and (36 < int(i_list[3].split('T')[1]) < 40):  # Day-end 1~45
                this_Series = all_data.loc[i]
                this_Series = this_Series.append(pd.Series([240], index=['Class'], name=i))
                result_data = result_data.append(this_Series)

    result_data.to_csv(path_or_buf=os.path.join(main_path, output_csv_path))
