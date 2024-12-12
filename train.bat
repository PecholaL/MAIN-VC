@echo off
setlocal

set "PYTHON_PATH=C:\Users\leeklll\miniconda3\envs\pyT\python.exe"

%PYTHON_PATH% "C:\Users\leeklll\Documents\DL\MAIN-VC\main.py" ^
    -c config.yaml ^
    -d c:\Users\leeklll\Documents\DL\mainVc_data ^
    -train_set train ^
    -train_index_file c:\Users\leeklll\Documents\DL\mainVc_data\train_samples_128.json ^
    -store_model_path c:\Users\leeklll\Documents\DL\mainVc_data\save\mainVcModel ^
    -log_dir c:\Users\leeklll\Documents\DL\mainVc_data\log ^
    -iters 200000 ^
    -summary_steps 5000

endlocal
pause