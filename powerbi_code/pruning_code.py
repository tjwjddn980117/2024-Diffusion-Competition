import pandas as pd
import numpy as np
from scipy import stats
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]

for csv_file in csv_file_list:
    print(csv_file)
    df = pd.read_csv(csv_file, encoding='cp949')
    init_df_len = len(df)
    z_scores = {}

    # 분포로 사용하지 않을 feature들과 사용할 feature들
    excluded_columns = ['가입자일련번호', '치아우식증유무','결손치유무','치아마모증유무','제3대구치(사랑니)이상','치석','시도코드','데이터공개일자','음주여부','기준년도','연령대코드(5세단위)','시력(좌)','시력(우)','흡연상태','구강검진 수검여부','데이터 기준일자']
    seleced_columns = [x for x in df.columns if x not in excluded_columns]
    # 신뢰구간을 9999.5%로 설정
    confidence_level = 0.99995

    # 분포로 나타내기 전에 먼저 NaN값을 가진 행들 제거
    df = df.dropna(subset=seleced_columns, axis=0)

    for column in df.columns:
        # excluded_codlumns는 사용하지 않겠다는 의지
        if column in excluded_columns:
            continue

        # 데이터 종류가 2개 이하면 분포를 구하면 안됨.
        # 표준편차가 구해질 가능성이 있는 환경 제거
        num_unique_values = df[column].nunique()
        if num_unique_values < 3:
            continue

        try:	
			# Z-점수를 계산
            df[column] = df[column].astype(float)
            z_scores[column] = np.abs(stats.zscore(df[column]))

        # data type float만 계산하게끔 예외처리
        except ValueError:
            pass
        except TypeError:
            pass

	# 이상치가 없는 행 찾기
    z_scores_list = list(z_scores.keys())
    non_outliers = np.ones(len(df), dtype=bool)  # 모든 행을 True로 초기화
    # 신뢰구간 컷으로 특이치 제거
    non_outliers = np.all([z_scores[column] < stats.norm.ppf(1 - (1 - confidence_level) / 2) for column in z_scores_list], axis=0)

	# 이상치 없는 행만 남겨 df 업데이트
    df = df[non_outliers]

	# 수정된CSV 파일로 저장.
	# 기본적으로 output_file은 따로 없이 파일이 그대로 수정이 됨.
	# 다른 파일로 output_file을 만들고 싶을 경우, output_file 변수 수정 요망.
    output_file = csv_file
    df.to_csv(output_file, index=False, encoding='cp949')