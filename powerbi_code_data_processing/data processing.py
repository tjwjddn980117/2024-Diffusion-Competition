import pandas as pd
import os

dir_path = os.getcwd()
print(dir_path)
excel_file_list = list()

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if '.CSV' in file:
            file_path = os.path.join(root,file)
            excel_file_list.append(file_path)
            print(file_path)
# 엑셀 파일을 읽어옵니다.
#excel_file = 'your_excel_file.xlsx'  # 파일 경로를 적절히 수정하세요.
for excel_file in excel_file_list:
    df = pd.read_excel(excel_file)

    # 5번째 열의 데이터 중에 '50'을 '49'로 바꿉니다.
    column_name = df.columns[4]  # 5번째 열의 열 이름
    df[column_name] = df[column_name].replace('50', '49')

    # 수정된 데이터프레임을 엑셀 파일로 저장합니다.
    #output_file = 'output_excel_file.xlsx'  # 저장할 파일 경로를 적절히 수정하세요.
    df.to_excel(excel_file, index=False)

    print(f"'{column_name}' 열의 '50'을 '49'로 변경하고 저장했습니다.")