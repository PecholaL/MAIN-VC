@echo off
setlocal enabledelayedexpansion

set "PYTHON_PATH=C:\Users\leeklll\miniconda3\envs\pyT\python.exe"

set "SOURCE_PATH=c:\Users\leeklll\Documents\DL\datasets\archive\VCTK-Corpus\VCTK-Corpus\wav48\p225\p225_001.wav"
set "TARGET_PATH=c:\Users\leeklll\Documents\DL\datasets\archive\VCTK-Corpus\VCTK-Corpus\wav48\p360\p360_011.wav"
set "OUTPUT_PATH=c:\Users\leeklll\Documents\DL\mainVc_data\inference"

%PYTHON_PATH% inference.py -s "!SOURCE_PATH!" -t "!TARGET_PATH!" -o "!OUTPUT_PATH!"

endlocal
pause