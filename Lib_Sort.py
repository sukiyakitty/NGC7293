import os


def files_sort_CD09(files):
    # CD09 sort
    files.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                              int(x.split('~')[0].split('-')[2]),
                              int(x.split('~')[1].split('_')[0].split('-')[1]) if (
                                      x.split('~')[1].split('_')[0].find('-') >= 0) else 0,
                              int(x.split('~T')[1].split('~C')[0]) if (x.find('~C') > 0) else int(
                                  x.split('~T')[1].split('.')[0]),
                              int(x.split('~C')[1].split('.')[0]) if (x.find('~C') > 0) else 0])
    return files


def files_sort_CD11(files):
    # CD11 sort
    # 2018-11-08~2018-11-08_1100_I-1_CD11~T24.jpg
    files.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                              int(x.split('~')[0].split('-')[2]),
                              int(x.split('~')[1].split('_')[1]),
                              int(x.split('~T')[1].split('.')[0])])
    return files


def files_sort_CD13(files):
    # CD13 sort
    # 2018-11-29~IPS-2_CD13~T10.jpg
    # 2018-12-03~I-2_CD13~T18.jpg
    # 2018-12-07~II-3_CD13~T2.jpg

    files.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                              int(x.split('~')[0].split('-')[2]),
                              int(x.split('~')[1].split('_')[0].split('-')[1]) if (
                                      x.split('~')[1].split('_')[0].find('-') >= 0) else 0,
                              int(x.split('~T')[1].split('~C')[0])
                              if (x.split('~T')[1].find('~C') >= 0) else int(x.split('~T')[1].split('.')[0])
                              ])
    return files


def files_sort_CD26(files):
    # CD26 sort
    # 2019-06-14~CD26_IPS(H9)~T1.png
    # 2019-06-14~CD26_STAGEI_0H~T1.png
    # 2019-06-17~CD26_STAGEII_IWR1~T14.png
    # 2019-06-19~CD26_STAGEII_D5~T5.png
    # 2019-06-27~CD26_End~T1.png
    # 2019-06-29~CD26_Result_IF~T1~C1.png
    files.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                              int(x.split('~')[0].split('-')[2]),
                              int(x.split('~')[1].split('_STAGEI_')[1].split('H')[0])
                              if (x.split('~')[1].find('_STAGEI_') >= 0) else
                              (-1 if (x.split('~')[1].find('_IPS(H9)') >= 0) else 72),
                              int(x.split('~T')[1].split('~C')[0])
                              if (x.split('~T')[1].find('~C') >= 0) else int(x.split('~T')[1].split('.')[0])])
    return files


def files_sort_CD27(files):
    # CD27 sort
    # 2019-06-22~CD27_IPS(H9)~T1.png
    # 2019-06-22~CD27_stageI_0h~T9.png
    # 2019-06-23~CD27_stageI_18h~T1.png
    # 2019-06-23~CD27_stageI_24h~T2.png
    # 2019-06-23~CD27_stageI_30h~T4.png
    # 2019-06-24~CD27_stageI_36h~T4.png
    # 2019-06-24~CD27_stageI_42h~T1.png
    # 2019-06-24~CD27_stageI_48h~T8.png
    # 2019-06-25~CD27_stageII_IWR1~T2.png
    # 2019-06-27~CD27_stageII_D5~T5.png
    # 2019-07-03~CD27_End~T1.png

    files.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                              int(x.split('~')[0].split('-')[2]),
                              int(x.split('~')[1].split('_stageI_')[1].split('h')[0])
                              if (x.split('~')[1].find('_stageI_') >= 0) else
                              (-1 if (x.split('~')[1].find('_IPS(H9)') >= 0) else 72),
                              int(x.split('~T')[1].split('~C')[0])
                              if (x.split('~T')[1].find('~C') >= 0) else int(x.split('~T')[1].split('.')[0])])
    return files


def files_sort_CD39(files):
    # CD39 sort
    # 2019-12-12~CD39_IPS_[]~T1.png
    # 2019-12-13~CD39_STAGE1_1H_[]~T2.png
    # 2019-12-13~CD39_STAGE1_18H_[]~T11.png
    # 2019-12-14~CD39_STAGE1_36H_[]~T3.png
    # 2019-12-15~CD39_STAGE1_64H_[]~T1.png

    files.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                              int(x.split('~')[0].split('-')[2]),
                              int(x.split('~')[1].split('_STAGE1_')[1].split('H')[0])
                              if (x.split('~')[1].find('_STAGE1_') >= 0) else
                              (-1 if (x.split('~')[1].find('_IPS_[]') >= 0) else 72),
                              int(x.split('~T')[1].split('~C')[0])
                              if (x.split('~T')[1].find('~C') >= 0) else int(x.split('~T')[1].split('.')[0])])
    return files


def files_sort_CD41(files):
    # CD41 sort
    # 2020-01-16~CD41_IPS~T1.png
    # 2020-01-16~CD41_STAGEI_1H~T1.png
    # 2020-01-17~CD41_STAGEI_39H~T16.png

    files.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                              int(x.split('~')[0].split('-')[2]),
                              int(x.split('~')[1].split('_STAGEI_')[1].split('H')[0])
                              if (x.split('~')[1].find('_STAGEI_') >= 0) else
                              (-1 if (x.split('~')[1].find('_IPS') >= 0) else 72),
                              int(x.split('~T')[1].split('~C')[0])
                              if (x.split('~T')[1].find('~C') >= 0) else int(x.split('~T')[1].split('.')[0])])
    return files


def files_sort_CD42(files):
    # CD42 sort
    # 2020-05-09~CD42_IPS-1~T1.png
    # 2020-05-12~CD42_Stage-1_1H~T1.png

    files.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                              int(x.split('~')[0].split('-')[2]),  # date
                              int(x.split('~')[1].split('_')[1].split('-')[1]) if (
                                      x.split('~')[1].split('_')[1].find('-') >= 0) else 0,
                              int(x.split('~')[1].split('_')[2].split('H')[0]) if (
                                      x.split('~')[1].find('H') >= 0) else 0,
                              int(x.split('~T')[1].split('~C')[0])
                              if (x.split('~T')[1].find('~C') >= 0) else int(x.split('~T')[1].split('.')[0])])
    return files


def files_sort_univers(files):
    # exp: CD46 sort
    # 2020-08-01~CD46_IPS-1~T22.png
    # 2020-08-02~CD46_Stage-1_1H~T12.png
    # 2020-08-03~CD46_Stage-1_24H~T1.png

    files.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                              int(x.split('~')[0].split('-')[2]),  # date
                              int(x.split('~')[1].split('_')[1].split('-')[1]) if (
                                      x.split('~')[1].split('_')[1].find('-') >= 0) else 0,  # -1 ; -2 ; -3
                              int(x.split('~')[1].split('_')[2].split('H')[0]) if (
                                      x.split('~')[1].find('H') >= 0) else 0,  # 1H ; 24H ; 48H
                              int(x.split('~T')[1].split('~C')[0])
                              if (x.split('~T')[1].find('~C') >= 0) else int(x.split('~T')[1].split('.')[0])])  # T1
    return files
