import pandas as pd
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file = 'NHIS2016.CSV'
csv_file_list = [file for file in file_list if file.endswith(".CSV")]

df = pd.read_csv(csv_file, encoding='cp949')
sum = 0
for csv_file in csv_file_list:
    print(csv_file)
    df = pd.read_csv(csv_file, encoding='cp949')
    print(len(df))
    print()
    sum+=len(df)
print(f'총합: {sum}')