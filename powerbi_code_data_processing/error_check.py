import pandas as pd


csv_file = 'NHIS2011.CSV'
problematic_column = '치아마모증유무'  # 에러가 발생한 열 이름으로 바꾸세요
df = pd.read_csv(csv_file, encoding='cp949')

valid_values = ['N', 'Y']
invalid_rows = df[~df['치아마모증유무'].isin(valid_values)]
if not invalid_rows.empty:
    print("열에 유효하지 않은 값을 포함하는 행이 있습니다:")
    print(invalid_rows)
print(df['치아마모증유무'][633171])