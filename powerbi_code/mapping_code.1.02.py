import pandas as pd
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]
csv_file_list = 'NHISwithConcat.CSV'

# CSV 파일 로드
df = pd.read_csv(csv_file_list, encoding='UTF-8')


# 연령대와 그룹 코드 맵핑 딕셔너리 생성
age_group_mapping = {
    1: "0~4세",
    2: "5~9세",
    3: "10~14세",
    4: "15~19세",
    5: "20~24세",
    6: "25~29세",
    7: "30~34세",
    8: "35~39세",
    9: "40~44세",
   10: "45~49세", 
   11:"50~54세", 
   12:"55~59세", 
   13:"60~64세", 
   14:"65~69세", 
  15:"70-74 세", 
   16:"75-79 세", 
  17:"80-84 세", 
  18:"85 세+"
}

# '연령대코드(5세단위)' 열의 값을 맵핑된 연령대로 변경
df['연령대코드(5세단위)'] = df['연령대코드(5세단위)'].map(age_group_mapping)

# cp949에러뜨면, encoding='UTF-8'로
csv_file = 'NHISwithConcatAGEstring.CSV'
df.to_csv(csv_file, index=False, encoding='cp949')
print(f'{csv_file} complete')

