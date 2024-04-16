import pandas as pd
import numpy as np
from scipy import stats
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]
#csv_file_list = ['NHIS2016.CSV']

for csv_file in csv_file_list:
    print(csv_file)
    df = pd.read_csv(csv_file, encoding='cp949')
    #print(df.shape)
    init_df_len = len(df)
    z_scores = {}
    rows_to_remove = set()
    #non_outliers = np.ones(len(df), dtype=bool)  # 모든 행을 True로 초기화
    #print(len(non_outliers))    

    excluded_columns = ['가입자일련번호', '치아우식증유무','결손치유무','치아마모증유무','제3대구치(사랑니)이상','치석','시도코드','데이터공개일자','음주여부','기준년도','연령대코드(5세단위)','시력(좌)','시력(우)','흡연상태','구강검진 수검여부','데이터 기준일자']
    seleced_columns = [x for x in df.columns if x not in excluded_columns]
    #seleced_columns = ['신장(5Cm단위)']
    # 신뢰구간을 99%로 설정
    confidence_level = 0.99995

    for column in df.columns:
        if column not in excluded_columns:
            # 특정 열에서 NaN 값을 포함하는 행 찾기
            rows_with_nan = df[df[column].isnull()]
            #print(f'{column}: {rows_with_nan.index}')
            # NaN 값을 포함하는 행들을 set으로 모아둠
            rows_to_remove.update(rows_with_nan.index)
    #print()
    #print(rows_to_remove)
    #for row in rows_to_remove:
    #    df = df.drop(labels=row, axis=0)
    df = df.dropna(subset=seleced_columns, axis=0)
    #print(len(df))
    #print(df)

    columns_with_nan = df.columns[df.isna().any()].tolist()
    #print("Columns with NaN values:", columns_with_nan)

    #for column in df.columns:
    #    print(len(df[column]))
    #input()
    copy_df = df.copy()
    for column in copy_df.columns:
        if column in excluded_columns:
            continue
        #print(column)
        num_unique_values = copy_df[column].nunique()
        #print(f'{column}의 데이터 종류: {num_unique_values}')

        if num_unique_values < 3: # 데이터 종류가 2개 이하면 나가게
            continue

        try:	
			# Z-점수를 계산
            copy_df[column] = copy_df[column].astype(float)
            #copy_df.dropna(subset=[column], inplace=True)
            #print(np.abs(stats.zscore(df[column])))
            z_scores[column] = np.abs(stats.zscore(copy_df[column]))
            #print(np.abs(stats.zscore(copy_df[column])))
            #print(z_scores[column])
            #print()
			#if column == '수축기혈압':
			#	print(df[column])
			#	print(f'{column} 의 zscore: {z_scores[column]}')
        except ValueError:
        			# data type float만 계산하게끔 예외처리
            pass
        except TypeError:
            pass
	# 이상치가 없는 행 찾기
    #print(stats.norm.ppf(1 - (1 - confidence_level) / 2))
    #input()
    #for col in seleced_columns:
    #    std_deviation = copy_df[col].std()
    #    print(std_deviation)

    #    print(copy_df[col].mean())
    #print(df)
    z_scores_list = list(z_scores.keys())
    #z_scores_list = seleced_columns
    #print(z_scores_list)
    #print(z_scores)
    #if '2016' in csv_file:
    #    for column in z_scores_list:
    #        print(column)
    #        print(z_scores[column])
    #print()
    #print('*******************')
    non_outliers = np.ones(len(copy_df), dtype=bool)  # 모든 행을 True로 초기화
    #print(non_outliers.shape)
    #print(len(non_outliers))    
    #print('*******************')
    #print()
    #input()
    #non_outliers = np.all([z_scores[column] < stats.norm.ppf(1 - (1 - confidence_level) / 2) for column in z_scores.keys()], axis=0)
    non_outliers = np.all([z_scores[column] < stats.norm.ppf(1 - (1 - confidence_level) / 2) for column in z_scores_list], axis=0)
    #non_outliers = np.all([z_scores[column] < stats.norm.ppf(1 - (1 - confidence_level) / 2) for column in z_scores_list])
    #print(f'non_outliers의 크기: {len(non_outliers)}')
    #print(non_outliers)
    #print()
    #print(f'copy_df의 크기: {len(copy_df)}')
    #print(copy_df)
    #print()
    #print(f'copy_df[non_outliers]의 크기: {len(copy_df[non_outliers])}')
    #print(copy_df[non_outliers])
    #print()
    #set_non_outliers = set(non_outliers)

    #columns_with_nan = copy_df.columns[copy_df.isna().any()].tolist()
    #print("Columns with NaN values:", columns_with_nan)
#
    #print()
	#for column in z_scores.keys():
	#	print(z_scores[column])
    #print(set_non_outliers)

	# 이상치 없는 행만 남겨 df 업데이트
    copy_df = copy_df[non_outliers]
    print(copy_df)
    #input()
    ##print(df)
    #print()

	# 수정된CSV 파일로 저장.
	# 기본적으로 output_file은 따로 없이 파일이 그대로 수정이 됨.
	# 다른 파일로 output_file을 만들고 싶을 경우, output_file 변수 수정 요망.
    output_file = csv_file
    #df.to_csv(output_file, index=False, encoding='cp949')