@echo off
cd /d f:\lyh\paddlespeech\papernoise\physic
D:\ProgramData\anaconda3\envs\noise\python.exe -u eval_v15.py > v15_eval_console.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL%
type v15_eval_console.txt
pause
