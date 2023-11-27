import opensmile
import scipy.io.wavfile as wav
import numpy as np
import os
import pandas as pd

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors, # 매 0.2초당 feature 를 기록하고, time window 를 0.1로 설정함
)

# feature set 과 level 조합에 대해서 알아보아야 함

# IITP 때는 아래와 같이 작성함.
# 1개의 wave 파일에 대해 88개의 feature 를 추출함
smile2 = opensmile.Smile(
        feature_set = opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


# print(smile.feature_names)

#실제 동작
# df = smile2.process_file('./voice/a_sound/230130_014430_7_SHG0130_5_A_Sound_0.wav')
# df2 = smile.process_file('./voice/dadada/230130_014701_7_SHG0130_8_DaDaDa_0.wav')
# df3 = smile.process_file('./voice/pataka/230130_014758_7_SHG0130_8_PaTaKa_0.wav')


# Set the directory containing CSV files
directory = "voice/e_sound"

# Get a list of CSV files in the directory
wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

# wav_files = ['230130_014701_7_SHG0130_8_DaDaDa_0.wav']
# wav_files = ['230130_014430_7_SHG0130_5_A_Sound_0.wav']

df = []
output =[]

for file in wav_files:
    print(file)
    filename = os.path.join(directory, file)
    
    result = smile2.process_file(filename)
    loudness = result['loudness_sma3_amean'].iloc[0]
    jitter_mean = result['jitterLocal_sma3nz_amean'].iloc[0]
    jitter_cv = result['jitterLocal_sma3nz_stddevNorm'].iloc[0]
    shimmer_mean = result['shimmerLocaldB_sma3nz_amean'].iloc[0]
    shimmer_cv = result['shimmerLocaldB_sma3nz_stddevNorm'].iloc[0]
    output.append([file, loudness, jitter_mean, jitter_cv, shimmer_mean, shimmer_cv])


out = pd.DataFrame(output, columns=['file', 'loudness', 'jitter_mean', 'jitter_cv', 'shimmer_mean', 'shimmer_cv'])
out.to_csv('./e_sound.csv')


# export df into csv
# df.to_csv('./a_sound_total.csv', index=True)

def calculate_decibels(audio_file):
    rate, data = wav.read(audio_file)
    # 정수형 데이터를 부동소수점 형식으로 변환
    data = data.astype(np.float32)
    # 스테레오 오디오 처리: 채널 별로 분리
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    # RMS(Root Mean Square) 값을 계산
    rms = np.sqrt(np.mean(np.square(data)))
    # 데시벨로 변환
    decibels = 20 * np.log10(rms / np.max(np.abs(data)))
    return decibels

# decibel_level = calculate_decibels('./voice/dadada/230130_014701_7_SHG0130_8_DaDaDa_0.wav')
# print(f"Decibel level: {decibel_level} dB")