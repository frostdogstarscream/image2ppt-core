@echo off
cd /d "%~dp0"
python services\product\launch.py --background
if errorlevel 1 pause
