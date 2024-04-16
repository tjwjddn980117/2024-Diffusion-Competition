import pandas as pd
import numpy as np
from scipy import stats
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]
# 빈 데이터프레임을 생성
dfs = list()

# 파일을 순회하며 데이터 전처리 및 병합
for csv_file in csv_file_list:
    print(csv_file)
	# cp949에러뜨면, encoding='UTF-8'로
    df = pd.read_csv(csv_file, encoding='cp949')
    print(df.keys())
    dfs.append(df)

print()
for csv_file in csv_file_list:
    print(csv_file)
	# cp949에러뜨면, encoding='UTF-8'로
    df = pd.read_csv(csv_file, encoding='cp949')

    # 비어 있는 열 식별
    empty_columns = df.columns[df.isna().all()]

    # 결과 출력
    print("비어 있는 열:")
    print(empty_columns)

#common_columns = set(dfs[0].columns)
#for df in dfs[1:]:
#    common_columns = common_columns.intersection(df.columns)
#
#for i in range(len(dfs)):
#    dfs[i] = dfs[i][list(common_columns)]
#