@echo off
setlocal enabledelayedexpansion

set "PYTHON_PATH=C:\Users\pecho\miniconda3\envs\pyT\python.exe"

set "RAW_DATA_DIR="
set "DATA_DIR="
set "SEGMENT_SIZE="
set "N_OUT_SPEAKERS="
set "TEST_PROP="
set "TRAINING_SAMPLES="
set "TESTING_SAMPLES="
set "N_UTT_ATTR="

for /f "tokens=1,2 delims==" %%A in (dataset_config.txt) do (
    if "%%A"=="raw_data_dir" set "RAW_DATA_DIR=%%B"
    if "%%A"=="data_dir" set "DATA_DIR=%%B"
    if "%%A"=="segment_size" set "SEGMENT_SIZE=%%B"
    if "%%A"=="n_out_speakers" set "N_OUT_SPEAKERS=%%B"
    if "%%A"=="test_prop" set "TEST_PROP=%%B"
    if "%%A"=="training_samples" set "TRAINING_SAMPLES=%%B"
    if "%%A"=="testing_samples" set "TESTING_SAMPLES=%%B"
    if "%%A"=="n_utt_attr" set "N_UTT_ATTR=%%B"
)

%PYTHON_PATH% "C:\Users\pecho\Documents\DL\MAIN-VC\data\preprocess\make_datasets.py" "%RAW_DATA_DIR%\wav48" "%RAW_DATA_DIR%\speaker-info.txt" "%DATA_DIR%" "%N_OUT_SPEAKERS%" "%TEST_PROP%" "%N_UTT_ATTR%"

%PYTHON_PATH% "C:\Users\pecho\Documents\DL\MAIN-VC\data\preprocess\sample_dataset.py" "%DATA_DIR%\train.pkl" "%DATA_DIR%\speaker2filenames.pkl" "%DATA_DIR%\train_samples_%SEGMENT_SIZE%.json" "%TRAINING_SAMPLES%" "%SEGMENT_SIZE%"

endlocal
pause