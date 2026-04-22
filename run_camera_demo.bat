@echo off
cd /d %~dp0
call .venv\Scripts\activate
set PYTHONPATH=%~dp0;%~dp0src;%~dp0src\controller;%~dp0src\util
python "%~dp0src\orbita_abh_camera.py"
pause
