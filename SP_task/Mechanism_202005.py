import os
import time
import shutil
import numpy as np
import pandas as pd


def select_gene(fpkm_norfile,template_file,output_name,template,shift = 0.2):

    if os.path.exists(fpkm_norfile):  # existed!
        fpkm_nor_df = pd.read_csv(fpkm_norfile, header=0, index_col=0)
        fpkm_nor_df = fpkm_nor_df.fillna(0)
        fpkm_nor_df = fpkm_nor_df.applymap(lambda x: float(x))
    else:
        pass

    if os.path.exists(template_file):  # existed!
        template_df = pd.read_csv(template_file, header=0, index_col=0)
        template_df = template_df.fillna(0)
        template_df = template_df.applymap(lambda x: float(x))
    else:
        pass

    out_df = fpkm_nor_df.copy()
    for each_gen in fpkm_nor_df.index:
        for each_condition in fpkm_nor_df.columns:
            each_value = fpkm_nor_df.loc[each_gen, each_condition]
            each_result = template_df.loc[template, each_condition]
            if each_value >= each_result - shift and each_value <= each_result + shift:
                out_df.loc[each_gen, each_condition + '_fit'] = 1
            else:
                out_df.loc[each_gen, each_condition + '_fit'] = 0

    out_df.to_csv(path_or_buf=output_name)

    return True


if __name__ == '__main__':

    fpkm_norfile = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\adw_processing\ALL_processing_max1_NO03_python_input.csv'
    template_file = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\adw_processing\result.csv'
    output_name = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\adw_processing\python_output_2.csv'
    # result=[0,0,0.894736842,0.315789474,0.105263158,1,0.105263158,0,0.947368421,0]
    # positive_file = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\adw_processing\positive.csv'
    # negative_file = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\adw_processing\negative.csv'

    output_name = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\adw_processing\positive_output_2.csv'
    select_gene(fpkm_norfile, template_file, output_name, 'positive', shift=0.2)
    output_name = r'C:\C137\Sub_Projects\202005_XGY_RNA-Seq\1+2\adw_processing\negative_output_2.csv'
    select_gene(fpkm_norfile, template_file, output_name, 'negative', shift=0.2)

