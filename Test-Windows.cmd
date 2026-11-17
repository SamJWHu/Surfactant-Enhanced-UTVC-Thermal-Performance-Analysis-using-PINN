@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto :python_missing
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv ".venv"
    if errorlevel 1 goto :python_missing
)
set "UTVC_PY=.venv\Scripts\python.exe"
"%UTVC_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
"%UTVC_PY%" -m utvc doctor
if errorlevel 1 goto :failed
"%UTVC_PY%" -m unittest discover -s tests -v
if errorlevel 1 goto :failed
"%UTVC_PY%" -m utvc evaluate --run examples\80W --out runs\windows-replay
if errorlevel 1 goto :failed
echo Finished. Open runs\windows-replay\T_C.png and pinn_metrics.csv.
pause
exit /b 0
:python_missing
echo Install 64-bit Python 3.12 and its py launcher from python.org, then rerun this file.
:failed
echo Test did not finish. Review the first error printed above.
pause
exit /b 1
