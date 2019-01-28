"""
We started preparing this file but it is not necessary since our challenge will
only run locally.

Script called when preparing a RAMP, either locally or on the backend.

Typically, it creates data/public_train.csv and data/public_test.csv
which will be committed into the repo and used by the starting kit,
and data/train.csv and data/test.csv which are kept locally.

It may also copy data/public_train.csv and data/public_test.csv
into the starting kit data/train.csv and data/test.csv.
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd

in_file = h5py.File('data/ASCAD.h5','r')
# Load profiling traces
X_train = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
# Load profiling labels
y_train = np.array(in_file['Profiling_traces/labels'])
# Load attacking traces
X_test = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
# Load attacking labels
y_test = np.array(in_file['Attack_traces/labels'])

df_train = pd.DataFrame(X_train)
df_test  = pd.DataFrame(Y_train)
