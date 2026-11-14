@echo off
REM Builds dist\RandomWalk.exe with PyInstaller inside a clean virtual environment,
REM so packages from your global Python (torch, tensorflow, pandas, ...) are not bundled.
REM Needs Python 3.9+ on PATH.
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment in .venv ...
    python -m venv .venv || goto :error
)
set PY=.venv\Scripts\python.exe

%PY% -m pip install --upgrade pip || goto :error
%PY% -m pip install numpy matplotlib pyinstaller || goto :error

%PY% -m PyInstaller --noconfirm --clean --onefile --windowed --name RandomWalk ^
    --hidden-import matplotlib.backends.backend_tkagg ^
    --exclude-module IPython --exclude-module jedi --exclude-module pandas ^
    --exclude-module scipy --exclude-module torch --exclude-module tensorflow ^
    --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --exclude-module PySide6 ^
    random_walk.py || goto :error

echo.
echo Done: dist\RandomWalk.exe
goto :eof

:error
echo Build failed.
exit /b 1
