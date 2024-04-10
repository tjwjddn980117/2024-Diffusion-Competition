import pandas as pd
import numpy as np
from scipy import stats
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]
result = pd.DataFrame()
result_len = 0
for csv_file in csv_file_list:
    print(csv_file)
    df = pd.read_csv(csv_file, encoding='cp949') 
    if '데이터공개일자' in df.columns:
        df = df.rename(columns={'데이터공개일자': '데이터 기준일자'})
    if '체중(5kg 단위)' in df.columns:
        df = df.rename(columns={'체중(5kg 단위)': '체중(5Kg 단위)'})
    if '치석유무' in df.columns:
        df = df.rename(columns={'치석유무': '치석'})
    if '2016' in csv_file:
        print(df)
        print()
    result = pd.concat([result, df], ignore_index=True, sort=False)
    result_len += len(df)

result.to_csv('NHISwithConcat.CSV', index=False)