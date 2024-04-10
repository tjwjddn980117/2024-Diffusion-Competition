import pandas as pd
import os

path = os.getcwd()
file_list = os.listdir(path)
csv_file_list = [file for file in file_list if file.endswith(".CSV")]
column_region = '시도코드'
column_sex = '성별코드'
column_age = '연령대코드(5세단위)'

mapping_region = {
    11: 'Seoul',
    26: 'Busan',
    27: 'Daegu',
    28: 'Incheon',
    29: 'Gwangju',
    30: 'Daejeon',
    31: 'Ulsan',
	36: 'Sejong',
    41: 'Gyeonggi-do',
    42: 'Gangwon-do',
    43: 'Chungcheongbuk-do',
    44: 'Chungcheongnam-do',
    45: 'Jeollabuk-do',
    46: 'Jeollanam-do',
    47: 'Gyeongsangbuk-do',
    48: 'Gyeongsangnam-do',
    49: 'Jeju-do'
}

mapping_sex = {
	1: 'male',
	2: 'female'
}


for csv_file in csv_file_list:
	
	# cp949에러뜨면, encoding='UTF-8'로
    df = pd.read_csv(csv_file, encoding='cp949')
	
    if '2011' in csv_file or '2012' in csv_file or '2013' in csv_file:
        df[column_age] = df[column_age] + 4

    df[column_region] = df[column_region].replace(50, 49)
    df[column_region] = df[column_region].map(mapping_region)

    df[column_sex] = df[column_sex].map(mapping_sex)

	# cp949에러뜨면, encoding='UTF-8'로
    df.to_csv(csv_file, index=False, encoding='cp949')
    print(f'{csv_file} complete')

    del df
