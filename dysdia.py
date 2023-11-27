import opensmile
import scipy.io.wavfile as wav
import numpy as np
import os
import pandas as pd
import scipy as sp
from scipy import signal

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors, # 매 0.2초당 feature 를 기록하고, time window 를 0.1로 설정함
)

# Set the directory containing CSV files
directory = "voice/dadada"

# Get a list of CSV files in the directory
wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

# wav_files = ['230130_014758_7_SHG0130_8_PaTaKa_0.wav']

df = []
output =[]

for file in wav_files:
    print(file)
    filename = os.path.join(directory, file)
    result = smile.process_file(filename)
    loudsignal = result['Loudness_sma3'].values
    peak_indices, peak_properties = sp.signal.find_peaks(loudsignal, height=0.5, distance=5)
    count = len(peak_indices)
    
    # print(peak_indices)
    #get standard deviation of peak_indices interval
    peak_indices_interval = np.diff(peak_indices)
    # print(peak_indices_interval)
    peak_indices_interval_std = np.std(peak_indices_interval)
    # print(peak_indices_interval_std)
    output.append([file, count, peak_indices_interval_std])

out = pd.DataFrame(output, columns=['file', 'count', 'interval variability'])
out.to_csv('./dadada.csv')


