import os
import time
import shutil
import numpy as np
import pandas as pd
from scipy import stats


def statistical_fluorescence_efficiency_by_chir(main_path, input_Experiment_Plan_csv):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    input_Experiment_Plan_csv = os.path.join(main_path, input_Experiment_Plan_csv)
    if not os.path.exists(input_Experiment_Plan_csv):
        print('!ERROR! The input_Experiment_Plan_csv does not existed!')
        return False

    inpu_csv = pd.read_csv(input_Experiment_Plan_csv, header=0, index_col=0)
    inpu_csv = inpu_csv.fillna(0)
    # inpu_csv = inpu_csv.applymap(lambda x: float(x))

    result = inpu_csv.groupby(['chir'])['IF_human'].mean()
    print(result)

    return True


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Statistical_Analysis.py !')

    main_path = r'C:\Users\Kitty\Desktop\Fractal\CD27'
    input_Experiment_Plan_csv = r'Experiment_Plan.csv'
    statistical_fluorescence_efficiency_by_chir(main_path, input_Experiment_Plan_csv)
