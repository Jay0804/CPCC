import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
# from Label_name import change_label
from tqdm import trange


if __name__ == "__main__":
    count = 0
    plt.figure(figsize=(13, 6))
    abs_path = f'D:/清华项目/CPCC会议论文'
    # file_name = abs_path + f'./imbal_data_csv_02/type(6)_num_off(480)/'
    file_name = abs_path + f'./imbal_data_csv_02/全周期/type(4)_num_off(256)/'
    # data_stream = np.array(pd.read_csv(file_name + f'HanPY02_data_off.csv', header=None))[:,1]
    data_stream = np.array(pd.read_csv(file_name + f'HanPY02_data_stream.csv', header=None))[:,1]
    x = np.arange(len(data_stream))

    plt.plot(x, data_stream, lw=2, ls='-', alpha=1)
    count += 1


    plt.legend(loc='best', fontsize=9, ncol=1)
    # plt.savefig(directory_path + f'/Results_{stream_condition}/RS_cbrt.pdf', bbox_inches='tight')
    plt.show()



