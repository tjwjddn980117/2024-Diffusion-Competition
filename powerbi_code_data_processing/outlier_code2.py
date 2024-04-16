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
	df = pd.read_csv(csv_file, encoding='cp949', nrows=34)
	#print(df)
	init_df_len = len(df)
	z_scores = {}	
	non_outliers = np.ones(len(df), dtype=bool)  # 모든 행을 True로 초기화
	
	excluded_columns = ['가입자일련번호', '치아우식증유무','결손치유무','치아마모증유무','제3대구치(사랑니)이상','치석','시도코드','데이터공개일자']
	seleced_columns = ['체중(5Kg 단위)', '허리둘레','총콜레스테롤', '트리글리세라이드']

	# 신뢰구간을 99%로 설정
	confidence_level = 0.99995
	

	for column in df.columns:
		if column in excluded_columns:
			continue
		try:	
			# Z-점수를 계산
			value_counts = df[column].value_counts()
			#print(value_counts)
			#print(value_counts)
			if value_counts.std() == 0:
				continue

			most_common_value = value_counts.idxmax()
			std_dev = value_counts.std()
			normal_distribution = stats.norm(loc=most_common_value, scale=std_dev)
			z_scores[column] = np.abs((df[column] - most_common_value) / std_dev)
			if column == '수축기혈압':
				print(df[column])
				print(most_common_value)
				print(f'{column} 의 zscore: {z_scores[column]}')
		except ValueError:
        			# data type float만 계산하게끔 예외처리
			pass
		except TypeError:
			pass
	# 이상치가 없는 행 찾기
	print(stats.norm.ppf(1 - (1 - confidence_level) / 2))
	non_outliers = np.all([z_scores[column] < stats.norm.ppf(1 - (1 - confidence_level) / 2) for column in z_scores.keys()], axis=0)
	set_non_outliers = set(non_outliers)
	#for column in z_scores.keys():
	#	print(z_scores[column])
	#print(set_non_outliers)

	# 이상치 없는 행만 남겨 df 업데이트
	df = df[non_outliers]
	#print(df)
	print()

	# 수정된CSV 파일로 저장.
	# 기본적으로 output_file은 따로 없이 파일이 그대로 수정이 됨.
	# 다른 파일로 output_file을 만들고 싶을 경우, output_file 변수 수정 요망.
	output_file = csv_file
	#df.to_csv(output_file, index=False, encoding='cp949')
