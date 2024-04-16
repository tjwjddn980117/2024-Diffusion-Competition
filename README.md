2024 AI 기반 의료 데이터 분석 경진대회

 - Classification
 - Stable Diffusion

Classification에 대한 코드들:
 - base code는 'Classification_for_base_Colab'에 있음. 
 - 우리가 레퍼런스한 코드는 'Classification_for_our_Colab'에 있음. 
  - Classification_for_our_Colab의 환경을 vscode같은 환경에서 구동할 경우 'Hackaton.bat'을 실행하거나 'Hackaton.txt'에 환경을 설치하면 된다. 
 - 데이터는 224*224의 diffusion mri 이미지를 사용하였다. 
  - Copy_of_pneumonia_detection_using_cnn_acc_99.ipynb의 f-score이 95.6 나왔다. 
 - data https://drive.google.com/drive/folders/1x0x9OFY30WPkbIAEtYIA68gHFMe7wOMr?usp=drive_link
Classification evaluation:
 - 'evaluation_intel' 폴더에 공모전에서 사용된 평가지표 exe 파일이 있다. 

diffusion에 대한 코드들:
 - https://github.com/lucidrains/denoising-diffusion-pytorch 에서 코드 레퍼런스함. 
 - 기본적으로 simple diffusion을 코드리뷰하였음.
 - 조금 더 디테일한 diffusion model은 'diffusion_for_detail_version' 폴더 안에 전체 모델이 있다. 
 - 개발 환경은 'requirement.bat'을 실행하거나, 'requirement.txt'에 있는 환경들을 설치하면 된다. 


PowerBi 데이터 전처리 코드:
 - 'powerbi_code_data_processing' 폴더에 이상치와 결측치를 제거하고, 행들의 이름이 달랐던 데이터들을 바로잡았다. 
 - pre-processed data https://drive.google.com/drive/folders/1TwQoqdR5j35k6BXhk3fdBO3ICI-3PrTo?usp=drive_link