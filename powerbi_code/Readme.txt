NHISwithMapping
패치내용

시도코드 
- 시도코드 영문화
- 세종시 추가

성별
- 성별 영문화

나이
- 나이 2014년 이후 기준으로 매핑

청력
- 청력(좌), 청력(우) 병합.

NHIS2011.CSV
- 이상치 열인 '제3대구치(사랑니)이상' 제거

NHIS2013.CSV
- 이상치 열인 column 34, column 35 제거

NHIS2016.CSV
- 다른 CSV파일들과 이름 다른 Column 수정 (띄어쓰기 수정)

NHIS2017.CSV
- 다른 CSV파일들과 이름 다른 Column 수정 (띄어쓰기 수정)

전체
- 모든 행이 NaN값으로 비어있는 Column들 제거

==================================================================

무지성으로 사용하실거면 NHISwithPruning99.995%.zip 풀어서 데이터 쓰셔도 됩니다.

신뢰구간 조정하고싶으시면
NHISwithMapping.zip 풀으시고,
pruning_code.py에 confidence_level 조정해서 사용하시면 됩니다.