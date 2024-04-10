import pandas as pd
import numpy as np
from scipy import stats
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]

for csv_file in csv_file_list:
    df = pd.read_csv(csv_file, encoding='cp949')
    init_df_len = len(df)
    z_scores = {}

    # 신뢰구간을 95%로 설정
    confidence_level = 0.995

    for column in df.columns:
        try:	
            if not df[column].isna().all():
                # 데이터 값의 출현 빈도수를 계산
                value_counts = df[column].value_counts()
                most_common_value = value_counts.idxmax()
                std_dev = value_counts.std()
            
                # 정규분포 근사화
                normal_distribution = stats.norm(loc=most_common_value, scale=std_dev)
            
                # Z-점수를 계산
                z_scores[column] = np.abs((df[column] - most_common_value) / std_dev)

        except ValueError:
        			# data type float만 계산하게끔 예외처리
            pass
        except TypeError:
            pass

	# 이상치가 없는 행 찾기
    non_outliers = np.all([z_scores[column] < stats.norm.ppf(1 - (1 - confidence_level) / 2) for column in z_scores.keys()], axis=0)

	# 이상치 없는 행만 남겨 df 업데이트
    df = df[non_outliers]

	# 수정된CSV 파일로 저장.
	# 기본적으로 output_file은 따로 없이 파일이 그대로 수정이 됨.
	# 다른 파일로 output_file을 만들고 싶을 경우, output_file 변수 수정 요망.
    output_file = csv_file
    df.to_csv(output_file, index=False, encoding='cp949')
