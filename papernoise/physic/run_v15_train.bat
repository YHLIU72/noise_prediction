@echo off
cd /d f:\lyh\paddlespeech\papernoise\physic
echo ========================================
echo V15 Training Started: %date% %time%
echo ========================================
D:\ProgramData\anaconda3\envs\noise\python.exe -u PIMBCN_train0713_v15.py > v15_train_log.txt 2>&1
echo.
echo Training finished or stopped: %date% %time%
pause
