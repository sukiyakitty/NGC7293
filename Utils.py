import os

import numpy as np
import pandas as pd
from scipy import stats


def temp_1(input_fpkm, output_fpkm, delta=1):
    if not os.path.exists(input_fpkm):
        print('!ERROR! The input CSV file does not existed!')
        return False

    pd_csv = pd.read_csv(input_fpkm, header=0, index_col=0)

    pd_la_csv = pd_csv.copy()
    pd_la_csv[['ips', 'm_10_24', 'm_8_36', 'm_6_48', 'm_10_48', 'm_6_24']] = pd_la_csv[
                                                                                 ['ips', 'm_10_24', 'm_8_36', 'm_6_48',
                                                                                  'm_10_48', 'm_6_24']] + 0.0000000001

    pd_fc_la_cs = pd_la_csv.copy()
    pd_fc_la_cs['m_10_24'] = pd_la_csv['m_10_24'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['m_8_36'] = pd_la_csv['m_8_36'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['m_6_48'] = pd_la_csv['m_6_48'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['m_10_48'] = pd_la_csv['m_10_48'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['m_6_24'] = pd_la_csv['m_6_24'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['ips'] = pd_la_csv['ips'] / pd_la_csv['m_8_36']

    pd_log2_fc_la_cs = pd_fc_la_cs.copy()
    pd_log2_fc_la_cs[['ips', 'm_10_24', 'm_8_36', 'm_6_48', 'm_10_48', 'm_6_24']] = np.log2(
        pd_log2_fc_la_cs[['ips', 'm_10_24', 'm_8_36', 'm_6_48', 'm_10_48', 'm_6_24']])

    pd_bool_log2_fc_la_cs = pd_log2_fc_la_cs.copy()
    pd_bool_log2_fc_la_cs['m_10_24'] = (np.abs(pd_log2_fc_la_cs['m_10_24']) <= delta)
    pd_bool_log2_fc_la_cs['m_6_48'] = (np.abs(pd_log2_fc_la_cs['m_6_48']) <= delta)
    pd_bool_log2_fc_la_cs['m_10_48'] = (np.abs(pd_log2_fc_la_cs['m_10_48']) >= delta)
    pd_bool_log2_fc_la_cs['m_6_24'] = (np.abs(pd_log2_fc_la_cs['m_6_24']) >= delta)
    pd_bool_log2_fc_la_cs['ips'] = (np.abs(pd_log2_fc_la_cs['ips']) >= delta)

    output_pd_nor_csv = pd_csv.copy()
    new_col_str = r'fold_change>=' + str(delta)
    output_pd_nor_csv[new_col_str] = pd_bool_log2_fc_la_cs[['m_10_24', 'm_6_48', 'm_10_48', 'm_6_24']].apply(
        lambda x: x.all(), axis=1)

    output_pd_nor_csv['temp_1'] = pd_bool_log2_fc_la_cs[['m_10_24', 'm_10_48', 'm_6_24']].apply(
        lambda x: x.all(), axis=1)

    output_pd_nor_csv['temp_2'] = pd_bool_log2_fc_la_cs[['m_6_48', 'm_10_48', 'm_6_24']].apply(
        lambda x: x.all(), axis=1)

    output_pd_nor_csv['new_'] = output_pd_nor_csv[['temp_1', 'temp_2']].apply(
        lambda x: x.any(), axis=1)

    # pd_nor_csv = pd_csv.copy()
    # pd_nor_csv[['ips', 'm_10_24', 'm_8_36', 'm_6_48', 'm_10_48', 'm_6_24']] = np.log2(
    #     pd_nor_csv[['ips', 'm_10_24', 'm_8_36', 'm_6_48', 'm_10_48', 'm_6_24']] + 0.0000000001)
    #
    # pd_nor_fc_csv = pd_nor_csv.copy()
    # pd_nor_fc_csv['m_10_24'] = pd_nor_csv['m_10_24'] / pd_nor_csv['m_8_36']
    # pd_nor_fc_csv['m_8_36'] = pd_nor_csv['m_8_36'] / pd_nor_csv['m_8_36']
    # pd_nor_fc_csv['m_6_48'] = pd_nor_csv['m_6_48'] / pd_nor_csv['m_8_36']
    # pd_nor_fc_csv['m_10_48'] = pd_nor_csv['m_10_48'] / pd_nor_csv['m_8_36']
    # pd_nor_fc_csv['m_6_24'] = pd_nor_csv['m_6_24'] / pd_nor_csv['m_8_36']
    # pd_nor_fc_csv['ips'] = pd_nor_csv['ips'] / pd_nor_csv['m_8_36']
    #
    # # fold change
    # pd_nor_fc_bool_csv = pd_nor_fc_csv.copy()
    # pd_nor_fc_bool_csv['m_10_24'] = (np.abs(pd_nor_fc_csv['m_10_24']) <= delta)
    # pd_nor_fc_bool_csv['m_6_48'] = (np.abs(pd_nor_fc_csv['m_6_48']) <= delta)
    # pd_nor_fc_bool_csv['m_10_48'] = (np.abs(pd_nor_fc_csv['m_10_48']) >= delta)
    # pd_nor_fc_bool_csv['m_6_24'] = (np.abs(pd_nor_fc_csv['m_6_24']) >= delta)
    # pd_nor_fc_bool_csv['ips'] = (np.abs(pd_nor_fc_csv['ips']) >= delta)
    # # fold change
    #
    # # sub fold change
    # pd_nor_fc_sub_csv = pd_nor_fc_csv.copy()
    # pd_nor_fc_sub_csv['m_10_24'] = pd_nor_fc_csv['m_10_24'] - pd_nor_fc_csv['m_8_36']
    # pd_nor_fc_sub_csv['m_8_36'] = pd_nor_fc_csv['m_8_36'] - pd_nor_fc_csv['m_8_36']
    # pd_nor_fc_sub_csv['m_6_48'] = pd_nor_fc_csv['m_6_48'] - pd_nor_fc_csv['m_8_36']
    # pd_nor_fc_sub_csv['m_10_48'] = pd_nor_fc_csv['m_10_48'] - pd_nor_fc_csv['m_8_36']
    # pd_nor_fc_sub_csv['m_6_24'] = pd_nor_fc_csv['m_6_24'] - pd_nor_fc_csv['m_8_36']
    # pd_nor_fc_sub_csv['ips'] = pd_nor_fc_csv['ips'] - pd_nor_fc_csv['m_8_36']
    # pd_nor_fc_sub_bool_csv = pd_nor_fc_sub_csv.copy()
    # pd_nor_fc_sub_bool_csv['m_10_24'] = (np.abs(pd_nor_fc_sub_csv['m_10_24']) <= delta)
    # pd_nor_fc_sub_bool_csv['m_6_48'] = (np.abs(pd_nor_fc_sub_csv['m_6_48']) <= delta)
    # pd_nor_fc_sub_bool_csv['m_10_48'] = (np.abs(pd_nor_fc_sub_csv['m_10_48']) >= delta)
    # pd_nor_fc_sub_bool_csv['m_6_24'] = (np.abs(pd_nor_fc_sub_csv['m_6_24']) >= delta)
    # pd_nor_fc_sub_bool_csv['ips'] = (np.abs(pd_nor_fc_sub_csv['ips']) >= delta)
    # sub fold change
    #
    # output_pd_nor_csv = pd_csv.copy()
    # new_col_str = r'fold_change>=' + str(delta)
    # output_pd_nor_csv[new_col_str] = pd_nor_fc_bool_csv[['m_10_24', 'm_6_48', 'm_10_48', 'm_6_24']].apply(
    #     lambda x: x.all(), axis=1)
    # output_pd_nor_csv['Significant'] = pd_nor_fc_sub_bool_csv[['m_10_24', 'm_6_48', 'm_10_48', 'm_6_24']].apply(
    #     lambda x: x.all(), axis=1)

    output_pd_nor_csv.to_csv(path_or_buf=output_fpkm)


def temp_2(input_fpkm, output_fpkm):
    if not os.path.exists(input_fpkm):
        print('!ERROR! The input CSV file does not existed!')
        return False

    pd_csv = pd.read_csv(input_fpkm, header=0, index_col=0)
    pd_la_csv = pd_csv.copy()
    pd_la_csv[['ips', 'm_10_24', 'm_8_36', 'm_6_48', 'm_10_48', 'm_6_24']] = pd_la_csv[
                                                                                 ['ips', 'm_10_24', 'm_8_36', 'm_6_48',
                                                                                  'm_10_48', 'm_6_24']] + 0.0000000001
    output_csv = pd_la_csv.copy()
    pd_rows = output_csv.shape[0]
    for i in range(pd_rows):
        this_index = output_csv.index[i]
        this_r1 = list(
            stats.ttest_1samp(pd_csv.iloc[i].loc[['m_10_24', 'm_8_36', 'm_6_48']], pd_csv.iloc[i].loc['m_10_48']))
        output_csv.loc[this_index, 'statistic_m_10_48'] = this_r1[0]
        output_csv.loc[this_index, 'pvalue_m_10_48'] = this_r1[1]
        this_r2 = list(
            stats.ttest_1samp(pd_csv.iloc[i].loc[['m_10_24', 'm_8_36', 'm_6_48']], pd_csv.iloc[i].loc['m_6_24']))
        output_csv.loc[this_index, 'statistic_m_6_24'] = this_r2[0]
        output_csv.loc[this_index, 'pvalue_m_6_24'] = this_r2[1]
        output_csv.loc[this_index, 'pvalue_sum'] = output_csv.iloc[i].loc['pvalue_m_10_48'] + output_csv.iloc[i].loc[
            'pvalue_m_6_24']

    output_csv.to_csv(path_or_buf=output_fpkm)


def temp_22(input_fpkm, output_fpkm):
    if not os.path.exists(input_fpkm):
        print('!ERROR! The input CSV file does not existed!')
        return False

    pd_csv = pd.read_csv(input_fpkm, header=0, index_col=0)

    output_csv = pd_csv.copy()
    pd_rows = output_csv.shape[0]
    for i in range(pd_rows):
        this_index = output_csv.index[i]
        # temp=stats.levene(pd_csv.iloc[i].loc[['m_10_24_log2', 'm_8_36_log2', 'm_6_48_log2']], pd_csv.iloc[i].loc[
        #     ['ips_log2', 'm_6_24_log2', 'RNA_12_24_log2', 'RNA_6_36_log2', 'RNA_12_36_log2', 'RNA_2_48_log2',
        #      'm_10_48_log2']])
        # print(temp)

        this_r1 = list(
            stats.ttest_ind(pd_csv.iloc[i].loc[['m_10_24_log2', 'm_8_36_log2', 'm_6_48_log2']], pd_csv.iloc[i].loc[
                ['ips_log2', 'm_6_24_log2', 'RNA_6_36_log2', 'RNA_2_48_log2']], equal_var=False))

        output_csv.loc[this_index, 't-test_statistic_Insufficient'] = this_r1[0]
        output_csv.loc[this_index, 't-test_pvalue_Insufficient'] = this_r1[1]

        this_r2 = list(
            stats.ttest_ind(pd_csv.iloc[i].loc[['m_10_24_log2', 'm_8_36_log2', 'm_6_48_log2']], pd_csv.iloc[i].loc[
                ['RNA_12_36_log2', 'm_10_48_log2']], equal_var=False))

        output_csv.loc[this_index, 't-test_statistic_over'] = this_r2[0]
        output_csv.loc[this_index, 't-test_pvalue_over'] = this_r2[1]

    output_csv.to_csv(path_or_buf=output_fpkm)


def temp_3(input_fpkm, output_fpkm, delta=[1], p=0.05):
    if not os.path.exists(input_fpkm):
        print('!ERROR! The input CSV file does not existed!')
        return False

    pd_csv = pd.read_csv(input_fpkm, header=0, index_col=0)
    output_all_csv = pd_csv.copy()
    pd_la_csv = pd_csv.copy()
    pd_la_csv[['ips', 'm_10_24', 'm_8_36', 'm_6_48', 'm_10_48', 'm_6_24']] = pd_la_csv[
                                                                                 ['ips', 'm_10_24', 'm_8_36', 'm_6_48',
                                                                                  'm_10_48', 'm_6_24']] + 0.0000000001

    # T-Test
    pd_rows = output_all_csv.shape[0]
    for i in range(pd_rows):
        this_index = output_all_csv.index[i]
        this_r1 = list(
            stats.ttest_1samp(pd_csv.iloc[i].loc[['m_10_24', 'm_8_36', 'm_6_48']], pd_csv.iloc[i].loc['m_10_48']))
        # output_all_csv.loc[this_index, 'statistic_m_10_48'] = this_r1[0]
        output_all_csv.loc[this_index, 'pvalue_m_10_48'] = this_r1[1]
        if this_r1[1] <= p:
            p_m_10_48 = True
        else:
            p_m_10_48 = False
        col_str_p_m_10_48 = r'pvalue_m_10_48<=' + str(p)
        output_all_csv.loc[this_index, col_str_p_m_10_48] = p_m_10_48

        this_r2 = list(
            stats.ttest_1samp(pd_csv.iloc[i].loc[['m_10_24', 'm_8_36', 'm_6_48']], pd_csv.iloc[i].loc['m_6_24']))
        # output_all_csv.loc[this_index, 'statistic_m_6_24'] = this_r2[0]
        output_all_csv.loc[this_index, 'pvalue_m_6_24'] = this_r2[1]
        if this_r2[1] <= p:
            p_m_6_24 = True
        else:
            p_m_6_24 = False
        col_str_p_m_6_24 = r'pvalue_m_6_24<=' + str(p)
        output_all_csv.loc[this_index, col_str_p_m_6_24] = p_m_6_24

        output_all_csv.loc[this_index, 'pvalue_sum'] = output_all_csv.iloc[i].loc['pvalue_m_10_48'] + \
                                                       output_all_csv.iloc[i].loc[
                                                           'pvalue_m_6_24']

    output_all_csv['pvalue_all<=' + str(p)] = output_all_csv[[col_str_p_m_10_48, col_str_p_m_6_24]].apply(
        lambda x: x.all(), axis=1)
    # T-Test

    # fold_change

    pd_fc_la_cs = pd_la_csv.copy()
    pd_fc_la_cs['m_10_24'] = pd_la_csv['m_10_24'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['m_8_36'] = pd_la_csv['m_8_36'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['m_6_48'] = pd_la_csv['m_6_48'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['m_10_48'] = pd_la_csv['m_10_48'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['m_6_24'] = pd_la_csv['m_6_24'] / pd_la_csv['m_8_36']
    pd_fc_la_cs['ips'] = pd_la_csv['ips'] / pd_la_csv['m_8_36']

    # pd_log2_fc_la_cs = pd_fc_la_cs.copy()
    output_all_csv[['log2_FC_ips', 'log2_FC_m_10_24', 'log2_FC_m_8_36', 'log2_FC_m_6_48', 'log2_FC_m_10_48',
                    'log2_FC_m_6_24']] = np.log2(
        pd_fc_la_cs[['ips', 'm_10_24', 'm_8_36', 'm_6_48', 'm_10_48', 'm_6_24']])

    # if not isinstance(delta, list):
    #     delta = [delta]
    #
    # for d in delta:
    #     fold_change_str = r'log2_fold_change>=' + str(d)
    #
    # pd_bool_log2_fc_la_cs = pd_log2_fc_la_cs.copy()
    # pd_bool_log2_fc_la_cs['m_10_24'] = (np.abs(output_all_csv['m_10_24']) <= delta)
    # pd_bool_log2_fc_la_cs['m_6_48'] = (np.abs(output_all_csv['m_6_48']) <= delta)
    # pd_bool_log2_fc_la_cs['m_10_48'] = (np.abs(output_all_csv['m_10_48']) >= delta)
    # pd_bool_log2_fc_la_cs['m_6_24'] = (np.abs(output_all_csv['m_6_24']) >= delta)
    # pd_bool_log2_fc_la_cs['ips'] = (np.abs(output_all_csv['ips']) >= delta)
    #

    # output_all_csv[fold_change_str] = pd_bool_log2_fc_la_cs[['m_10_24', 'm_6_48', 'm_10_48', 'm_6_24']].apply(
    #     lambda x: x.all(), axis=1)
    #
    # fold_change_str['temp_1'] = pd_bool_log2_fc_la_cs[['m_10_24', 'm_10_48', 'm_6_24']].apply(
    #     lambda x: x.all(), axis=1)
    #
    # fold_change_str['temp_2'] = pd_bool_log2_fc_la_cs[['m_6_48', 'm_10_48', 'm_6_24']].apply(
    #     lambda x: x.all(), axis=1)
    #
    # fold_change_str['new_'] = fold_change_str[['temp_1', 'temp_2']].apply(
    #     lambda x: x.any(), axis=1)
    # fold_change

    # save
    output_all_csv.to_csv(path_or_buf=output_fpkm)
    return True


def temp_4(input_1, input_2, output_fpkm):
    df_1 = pd.read_csv(input_1, header=0, index_col=0)
    df_2 = pd.read_csv(input_2, header=0, index_col=0)

    df_join = pd.merge(df_1, df_2, how='outer', on=None, left_on=None, right_on=None, left_index=True, right_index=True,
                       sort=True,
                       suffixes=('_x', '_y'), copy=True, indicator=False)

    df_join.to_csv(path_or_buf=output_fpkm)


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Utils.py !')

    # ---the pandas display settings---
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100000)
    pd.set_option('display.width', 100000)
    np.set_printoptions(threshold=None)

    # input_fpkm = r'C:\Users\Kitty\Desktop\RNA-Seq\test\test.csv'
    # output_fpkm = r'C:\Users\Kitty\Desktop\RNA-Seq\test\output_test.csv'
    # temp_1(input_fpkm, output_fpkm)
    # temp_2(input_fpkm, output_fpkm)

    # input_fpkm = r'C:\Users\Kitty\Desktop\RNA-Seq\FPKM.csv'
    # output_fpkm = r'C:\Users\Kitty\Desktop\RNA-Seq\output.csv'
    # temp_3(input_fpkm, output_fpkm, delta=1, p=0.05)
    # temp_2(input_fpkm, output_fpkm)

    # input_1 = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\1.csv'
    # input_2 = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\2.csv'
    # output_fpkm = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\ALL.csv'
    # temp_4(input_1, input_2, output_fpkm)

    input_fpkm = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\t-test_input.csv'
    output_fpkm = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\t-test_output.csv'
    temp_22(input_fpkm, output_fpkm)
