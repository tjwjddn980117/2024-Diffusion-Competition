import pandas as pd
import numpy as np
from scipy import stats
import os

# '청력'이라는 새로운 열을 만들기 위한 열의 순서 변경 함수
def move_column(df, col_name, position):
    col = df.pop(col_name)
    df.insert(position, col_name, col)

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]

for csv_file in csv_file_list:
    df = pd.read_csv(csv_file, encoding='cp949') 
    
    #띄어쓰기 제거
    df.columns = df.columns.str.strip()

    # NHIS2011의 이상한 것 제거
    if '2011' in csv_file:
        df = df.drop(['제3대구치(사랑니)이상'], axis=1)

    # NHIS2013의 이상한 것 제거
    #if '2013' in csv_file:
    #    df = df.iloc[:, :-2]

    # NHIS2016의 이상한 것 변경
    if '2016' in csv_file:
        df = df.rename(columns={'체중(5kg단위)': '체중(5kg 단위)'})
    
    # NHIS2017의 이상한 것 변경
    if '2017' in csv_file:
        df = df.rename(columns={'체중(5kg단위)': '체중(5kg 단위)'})
        df = df.rename(columns={'구강검진수검여부': '구강검진 수검여부'})

    # 결측치를 제거합니다.
    df = df.dropna(axis=1, how='all')

    # 청력 수정한 것
    # 청력(좌)와 청력(우)가 모두 1인 행을 1로 설정, 그 외는 2로 설정
    df['청력'] = np.where((df['청력(좌)'] == 1) & (df['청력(우)'] == 1), 1, 2)
    # 청력(좌)와 청력(우) 열 삭제
    df = df.drop(['청력(좌)', '청력(우)'], axis=1)

    # '청력' 열을 원하는 위치에 위치
    move_column(df, '청력', 10)

    # 수정된CSV 파일로 저장.
	# 기본적으로 output_file은 따로 없이 파일이 그대로 수정이 됨.
	# 다른 파일로 output_file을 만들고 싶을 경우, output_file 변수 수정 요망.
    output_file = csv_file
    df.to_csv(output_file, index=False, encoding='cp949')
