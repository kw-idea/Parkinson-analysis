import numpy as np
import pandas as pd
import scipy as sp
import os
from datetime import datetime

# 눈 깜빡임 감지 함수
def detect_blink(data, threshold):
    binary_data = (data < threshold).astype(int)
    blink_signal = np.diff(binary_data)
    blink_number = np.sum(blink_signal > 0)
    blink_info = np.where(blink_signal > 0)
    return blink_number, blink_info

# 초당 눈 깜빡임 횟수 계산 함수
def calculate_blinks_per_second(blink_number, total_time):
    if total_time <= 0:
        return 0
    return blink_number / total_time


# 시간 변환 함수
def convert_to_timestamp(date_str):
    # 분리된 값들을 int로 변환
    data_parts = date_str.split(';')

    year, month, day, hour, minute = map(int, data_parts[:5])
    second = 0
    microsecond = '0'
    # 예상하는 데이터 개수가 5개라면 - 초와 마이크로초는 0으로 설정
    if len(data_parts) == 5:
        second = 0
        microsecond = 0
    elif len(data_parts) == 6: # 데이터 개수가 6개라면 마이크로초는 0으로 설정
        second = int(data_parts[5])
        microsecond = 0
    elif len(data_parts) == 7: # 데이터 개수가 7개라면 마이크로초를 설정
        second = int(data_parts[5])
        microsecond = microsecond.zfill(9)
        microsecond = int(data_parts[6][:-3])

    dt = datetime(year, month, day, hour, minute, second, microsecond)

    # timestamp로 변환
    timestamp = dt.timestamp()
    
    return timestamp

# Set the directory containing CSV files
directory = "blink"

# Get a list of CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
# csv_files = ['230130_023149_8_MYH0130_7_Screen_Gaze.csv']

df = []
output=[]
for file in csv_files:
    print(file)
    filename = os.path.join(directory, file)
    df = pd.read_csv(filename)
    
    df["timestamp"] = df["time"].apply(convert_to_timestamp)
    # print(convert_to_timestamp('2023;1;30;11;30'))
    # 눈 깜빡임 감지
    #eye_open = df['rightEyeOpen']
    eye_open = df['leftEyeOpen']
    blink_number, blink_info = detect_blink(eye_open, threshold=0.1)
    
    # 데이터 처리 시간 측정
    total_time = df['timestamp'].max() - df['timestamp'].min()
    
    
    # 데이터가 충분하지 않으면 분석을 수행하지 않음
    if total_time > 0 and blink_number >= 3:
        # 전체 데이터에 대한 표준편차 계산
        average_blink_interval_variability = np.std(np.diff(df['timestamp']))
        
        # 초당 눈 깜빡임 횟수 계산
        blink_frequency = calculate_blinks_per_second(blink_number, total_time)

        # 처음과 마지막 눈 깜빡임 지점 찾기
        blink_tuple = blink_info[0]
        first_blink_index = blink_tuple[0]
        last_blink_index = blink_tuple[-1]

        # 해당 인덱스에 대한 timestamp 값 얻기
        start_time = df['timestamp'].iloc[first_blink_index]
        end_time = df['timestamp'].iloc[last_blink_index]

        # 처음과 마지막 눈깜빡임 사이의 시간을 5등분
        interval_length = (end_time - start_time) / 5
        blink_interval_variability_by_time = [] 

        for i in range(5):
            interval_start = start_time + i * interval_length
            interval_end = interval_start + interval_length

            # 등분별 대한 표준편차 계산
            data_range = df[(df['timestamp'] >= interval_start) & (df['timestamp'] <= interval_end)]
            interval_variability = np.std(np.diff(data_range['timestamp']))
            blink_interval_variability_by_time.append(interval_variability)

        output.append([file, total_time, blink_number, blink_frequency, average_blink_interval_variability, blink_interval_variability_by_time[0], blink_interval_variability_by_time[1], blink_interval_variability_by_time[2], blink_interval_variability_by_time[3], blink_interval_variability_by_time[4]])
    else:
        output.append([file, total_time, blink_number, 0, 0, 0])

#file output
out = pd.DataFrame(output, columns=['file', 'time', 'blink_number', 'blink_frequency', 'average_blink_interval_variability', 'blink_interval_variability_by_time 1', 'blink_interval_variability_by_time 2', 'blink_interval_variability_by_time 3', 'blink_interval_variability_by_time 4', 'blink_interval_variability_by_time 5'])
csv_filename = 'blink.csv'
out.to_csv(csv_filename, index=False)
