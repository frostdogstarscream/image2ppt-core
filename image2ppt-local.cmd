@echo off
setlocal
"%~dp0.runtime\core-venv\Scripts\python.exe" -X utf8 "%~dp0services\windows\local_cli.py" %*
exit /b %errorlevel%
