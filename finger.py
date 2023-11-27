import numpy as np
import pandas as pd
import scipy as sp
import os
from datetime import datetime

# 정상 태핑 count 함수
def count_tap_directions(data):
    tap_number = 0
    prev_direction = None
    tap_indices = []  # 태핑의 인덱스를 저장할 리스트

    for index, item in enumerate(data):
        direction = item.strip().split(" ")[0]

        if direction != prev_direction:
            tap_number += 1
            tap_indices.append(index)  # 태핑 인덱스를 리스트에 추가

        prev_direction = direction

    return tap_number, tap_indices

# 초당 정상 태핑 횟수 계산 함수
def calculate_taps_per_second(tap_number, total_time):
    if total_time <= 0:
        return 0
    return tap_number / total_time

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
directory = "finger"

# Get a list of CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
#csv_files = ['230210_064429_31_CDH0210_1_Finger.csv']

df = []
output=[]
for file in csv_files:
    print(file)
    filename = os.path.join(directory, file)
    df = pd.read_csv(filename)
    
    df["timestamp"] = df["time"].apply(convert_to_timestamp)

    # 데이터 처리 시간 측정
    total_time = df['timestamp'].max() - df['timestamp'].min()

    # 정상 태핑 이벤트 발생 수
    tap_directions = df["direction"].tolist()
    tap_number, tap_indices = count_tap_directions(tap_directions)

    # 초당 정상 태핑 이벤트 횟수 계산
    tap_frequency = calculate_taps_per_second(tap_number, total_time)

    # 전체 데이터에 대한 표준편차 계산
    Average_tap_interval_variability = np.std(np.diff(df['timestamp']))

    tap_interval_variability_by_time = []

    # 처음과 마지막 정상 태핑 이벤트 지점 찾기
    #tap_tuple = tap_indices[0]
    first_tap_index = tap_indices[0]
    last_tap_index = tap_indices[-1]

    # 해당 인덱스에 대한 timestamp 값 얻기
    start_time = df['timestamp'].iloc[first_tap_index]
    end_time = df['timestamp'].iloc[last_tap_index]

    # 처음과 마지막 태핑 사이의 시간을 5등분
    interval_length = (end_time - start_time) / 5
    interval_variability_by_time = [] 

    for i in range(5):
        interval_start = start_time + i * interval_length
        interval_end = interval_start + interval_length

        # 등분별 대한 표준편차 계산
        data_range = df[(df['timestamp'] >= interval_start) & (df['timestamp'] <= interval_end)]
        interval_variability = np.std(np.diff(data_range['timestamp']))
        tap_interval_variability_by_time.append(interval_variability)

    output.append([file, total_time, tap_number, tap_frequency, Average_tap_interval_variability, tap_interval_variability_by_time[0], tap_interval_variability_by_time[1], tap_interval_variability_by_time[2], tap_interval_variability_by_time[3], tap_interval_variability_by_time[4]])

#file output
out = pd.DataFrame(output, columns=['file', 'time', 'tap_number', 'tap frequency', 'Average tap interval variability', 'tap interval variability by time1', 'tap interval variability by time2', 'tap interval variability by time3', 'tap interval variability by time4', 'tap interval variability by time5'])
csv_filename = 'finger_data.csv'
out.to_csv(csv_filename, index=False)
