import pandas as pd
import pandas as pd
import numpy as np
from scipy import stats
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]

for csv_file in csv_file_list:
    print(csv_file)
    df = pd.read_csv(csv_file, encoding='cp949', nrows=34)
    mixed_type_columns = []
    for column in df.columns:
        if df[column].apply(type).nunique() > 1:
            #if column == 'Unnamed: 35':
            #    print(column)
            mixed_type_columns.append(column)

    print("열에서 데이터 타입이 혼합된 열:", mixed_type_columns)

    ## 행마다 데이터 타입 확인
    #mixed_type_rows = []
    #for index, row in df.iterrows():
    #    if row.apply(type).nunique() > 1:
    #        mixed_type_rows.append(index)
    #
    #print("행에서 데이터 타입이 혼합된 행:", mixed_type_rows)
