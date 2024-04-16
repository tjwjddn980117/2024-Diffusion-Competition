import pandas as pd
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file = 'NHIS2011.CSV'
csv_file_list = [file for file in file_list if file.endswith(".CSV")]

df = pd.read_csv(csv_file, encoding='cp949')

item_set = set()
# 맨 마지막 2열을 제외한 데이터프레임 생성
for item in df['제3대구치(사랑니)이상']:
    item_set.add(item)
item_set = set(item_set)
print(len(item_set))
print(item_set)

item_set2 = set()
for item in df['치석']:
    item_set2.add(item)

print(len(item_set2))