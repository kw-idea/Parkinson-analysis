# Step Tracker 코드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import os
from scipy import signal
from datetime import datetime

def convert_to_timestamp(date_str):
    # 분리된 값들을 int로 변환
    data_parts = date_str.split(';')

    year, month, day, hour, minute, second = map(int, data_parts[:6])
    microsecond = 0
    # 예상하는 데이터 개수가 7개인지 확인
    if len(data_parts) != 7:
        microsecond = 0
    else:
        microsecond = int(data_parts[6][:-3])

    dt = datetime(year, month, day, hour, minute, second, microsecond)

    # timestamp로 변환
    timestamp = dt.timestamp()
    return timestamp


# Set the directory containing CSV files
directory = "gait"

# Get a list of CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# csv_files = ['230518_104314_HJY0518_66_Gait.csv']

df = []
output=[]
for file in csv_files:
    print(file)
    filename = os.path.join(directory, file)
    df = pd.read_csv(filename)

    # Calculate the magnitude of acceleration
    #df['magnitude'] = np.sqrt(df['xaccelerationMinusGx']**2 + df['yaccelerationMinusGy']**2 + df['zaccelerationMinusGz']**2)
    
    # just use zaccelerationMinusGz
    df['magnitude'] = df['zaccelerationMinusGz']
    
    df["timestamp"] = df["time"].apply(convert_to_timestamp)

    # 초당 몇번 계측이 되었는가...
    sampling_rate = df['timestamp'].size/(df['timestamp'].max() - df['timestamp'].min())

    # detrended - magnitude 를 활용할 때에는 필수적으로 사용해야함. z 만 사용할 때에는 괜찮음.
    # mag_detrended = sp.signal.detrend(df['magnitude'])

    # butterfilter
    filter_order = 1 
    low_pass_cutoff_freq = 2 # 2Hz 이상의 주파수는 제거함. 
    sos = signal.butter(filter_order, low_pass_cutoff_freq, 'lowpass', fs=sampling_rate, output='sos')
    mag_filtered = signal.sosfilt(sos, df['magnitude'])

    # find peaks 의 parameter 정의
    min_distance_between_peaks = 3 if sampling_rate < 40 else 0.3 * sampling_rate 
    min_peak_height = 0 # this threshold should be "learned" based on analysis of test data
    max_peak_height = 30 # same with this threshold

    peak_indices, peak_properties = sp.signal.find_peaks(mag_filtered, height=min_peak_height, distance=min_distance_between_peaks)

    # filter out peaks that exceed our maximum
    filtered_peak_indices = []
    for peak_index in peak_indices:
        if(mag_filtered[peak_index] <= max_peak_height):
            filtered_peak_indices.append(peak_index)
        else:
            print("[exceed max] elemintated peak index {} with value {}".format(peak_index, mag_filtered[peak_index]))

    peak_indices = list(filtered_peak_indices)

    # peaks must come in at least pairs and be within max_distance from each other
    max_distance_between_peaks = 3 * sampling_rate # 3초 이내에 발생한 peak 만을 사용함.
    filtered_peak_indices2 = set()
    for i in range(0, len(peak_indices) - 1):
        peak_index1 = peak_indices[i]
        peak_index2 = peak_indices[i + 1]
        if(peak_index2 - peak_index1 < max_distance_between_peaks):
            filtered_peak_indices2.add(peak_index1)
            filtered_peak_indices2.add(peak_index2)
        else:
            print("[within max distance] Eliminated peak index {} with value {}".format(peak_index1, mag_filtered[peak_index1]))

    peak_indices = sorted(list(filtered_peak_indices2))

    # calculate metrics
    total_time = df['timestamp'].max() - df['timestamp'].min()
    step_time = (df['timestamp'][max(peak_indices)]-df['timestamp'][min(peak_indices)])
    
    num_steps =0
    avg_cadence = 0
    if sampling_rate > 40:
        num_steps = len(peak_indices)
        avg_cadence = (num_steps -1) * 60 / (total_time)
    else: # if 기존 5Hz 설정에서는 step 에 x 2 해주어야함 
        num_steps = len(peak_indices) * 2
        avg_cadence = (num_steps -1) * 60 / (total_time)
    
    # cadence 의 값들 구하기
    interval_time_values = 0
    if sampling_rate > 40:
        interval_time_values = [ 60/(df['timestamp'][peak_indices[i + 1]]-df['timestamp'][peak_indices[i]]) for i in range(len(peak_indices) - 1)]
    else:
        interval_time_values = [ 120/(df['timestamp'][peak_indices[i + 1]]-df['timestamp'][peak_indices[i]]) for i in range(len(peak_indices) - 1)]

    # peak index 값들 로그로 출력
    #print(interval_time_values)

    output.append([file, num_steps, avg_cadence, np.std(interval_time_values), 80/num_steps, total_time])

#file output
out = pd.DataFrame(output, columns=['file', 'steps', 'cadence', 'cadence variability', 'step length', 'time' ])
csv_filename = 'gait.csv'
out.to_csv(csv_filename)