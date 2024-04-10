import pandas as pd
import numpy as np
from scipy import stats
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]
csv_file_list = ['NHIS2016.CSV']

for csv_file in csv_file_list:
    print(csv_file)
    df = pd.read_csv(csv_file, encoding='cp949')
    print(df.shape)
    init_df_len = len(df)
    z_scores = {}	
    
    excluded_columns = ['가입자일련번호', '치아우식증유무','결손치유무','치아마모증유무','제3대구치(사랑니)이상','치석','시도코드','데이터공개일자','음주여부','기준년도','연령대코드(5세단위)','시력(좌)','시력(우)','흡연상태','구강검진 수검여부','데이터 기준일자']
    seleced_columns = ['신장(5Cm단위)']
    
    confidence_level = 0.99995

    for column in df.columns:
        if column not in excluded_columns:
            rows_with_nan = df[df[column].isnull()]
            rows_to_remove = set(rows_with_nan.index)
    
    df = df.drop(rows_to_remove)
    copy_df = df.copy()
    
    for column in copy_df.columns:
        if column in excluded_columns:
            continue
        
        num_unique_values = copy_df[column].nunique()
        
        if num_unique_values < 3:
            continue

        try:
            copy_df[column] = copy_df[column].astype(float)
            copy_df.dropna(subset=[column], inplace=True)
            z_scores[column] = np.abs(stats.zscore(copy_df[column]))
        except ValueError:
            pass
        except TypeError:
            pass

    z_scores_list = list(z_scores.keys())
    
    non_outliers = np.all([z_scores[column] < stats.norm.ppf(1 - (1 - confidence_level) / 2) for column in z_scores_list], axis=0)

    copy_df = copy_df[non_outliers]
    
    output_file = csv_file
    #df.to_csv(output_file, index=False, encoding='cp949')