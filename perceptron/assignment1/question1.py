import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_dir = "./Datasets-Question1"
tr_data_str = "./Datasets-Question1/dataset{}/Test{}.csv"
ts_data_str = "./Datasets-Question1/dataset{}/Test{}.csv"

num_dataset = len(next(os.walk(data_dir))[1])

for i in range(1, num_dataset+1):
    tr_data_file = tr_data_str.format(i, i)
    ts_data_file = ts_data_str.format(i, i)

    print("Training data file: {}".format(tr_data_file))
    print("Test data file: {}".format(ts_data_file))
