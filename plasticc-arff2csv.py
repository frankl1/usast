import time 
import os 
import numpy as np
import pandas as pd
import argparse

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from sast.utils import *

parser = argparse.ArgumentParser()

parser.add_argument('dataset_folder', help='The folder containing the datasets')
parser.add_argument('dataset_name', help='The name of the dataset, the part before the _[dimension]')
parser.add_argument('-b', '--bands', help='String of dimensions to consider', default='giruyz')

args = parser.parse_args()

dataset_folder = args.dataset_folder
dataset_name = args.dataset_name
bands = args.bands

for b in bands:
    print('#########> Reading band:', b)
    dname = f'{dataset_name}_{b}'
    train_ds, train_noise_ds, test_ds, test_noise_ds = load_uncertain_dataset(dataset_folder, dname)

    train_ds[train_ds.columns[-1]] = train_ds[train_ds.columns[-1]].astype((int))
    test_ds[test_ds.columns[-1]] = test_ds[test_ds.columns[-1]].astype((int))

    train_ds.to_csv(os.path.join(dataset_folder, dname, f'{dname}_TRAIN.csv'))
    test_ds.to_csv(os.path.join(dataset_folder, dname, f'{dname}_TEST.csv'))

    train_noise_ds.to_csv(os.path.join(dataset_folder, dname, f'{dname}_NOISE_TRAIN.csv'))
    test_noise_ds.to_csv(os.path.join(dataset_folder, dname, f'{dname}_NOISE_TEST.csv'))