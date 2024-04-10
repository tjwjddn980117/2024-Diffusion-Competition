rem you should change the root with your own environment path root.
rem and you could change the ENV_NAME with your one vitual environment.
set root=C:\Users\qowor\anaconda3
set ENV_NAME=Diffusion

if not exist "%root%" (
    echo check the root please: %root%
    pause
    exit
)

call %root%\Scripts\activate.bat %root%

echo make the virtual environment '%ENV_NAME%'
call conda create -y -n %ENV_NAME% python=3.10

echo enter the virtual environment.
call conda activate %ENV_NAME%

echo start downloading environment for %ENV_NAME%.
call conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
call conda install -y conda-forge::packaging conda-forge::tqdm conda-forge::einops conda-forge::accelerate
call pip install ema-pytorch
call pip install pytorch-fid

call conda deactivate

echo complete. 