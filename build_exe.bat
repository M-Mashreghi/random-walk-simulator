@echo off
REM Builds RandomWalk.exe (Windows) with PyInstaller. Needs Python 3.9+ on PATH.
cd /d "%~dp0"
python -m pip install --upgrade numpy matplotlib pyinstaller || goto :error
python -m PyInstaller --noconfirm --onefile --windowed --name RandomWalk random_walk.py || goto :error
echo.
echo Done: dist\RandomWalk.exe
goto :eof
:error
echo Build failed.
exit /b 1
