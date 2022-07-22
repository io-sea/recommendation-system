#%% Load real data
import numpy as np
from scipy import signal
import pickle
import sympy
import burst_detection as bd
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.getcwd())
with open('C:\\Users\\a770398\\IO-SEA\\io-sea-3.4-analytics\\jobmodel\\jobmodel\\job_data.pkl', 'rb') as fp:
    data = pickle.load(fp)
jobids = list(data.keys())
j = 12
R = data[jobids[j]]['bytesRead']
W = data[jobids[j]]['bytesWritten']


def burst_detect(W):
    signal = pd.Series(W)
    d= np.ones(shape=signal.shape)*np.max(signal)
    n = len(signal)
    [q, _, _, p] = bd.burst_detection(signal, d, len(signal), 1.5, 2,smooth_win=2)
    bursts = bd.enumerate_bursts(q, 's='+str(1.5)+', g='+str(1.0))
    bursts = bd.burst_weights(bursts, signal, d, p)
    bursts.sort_index(inplace=True)
    return bursts


if __name__=="__main__":
    signal = W[250:450]
    bursts = burst_detect(signal)
    
    plt.rcParams["figure.figsize"] = (10,5)
    print(bursts)