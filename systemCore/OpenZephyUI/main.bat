@echo off

REM --- Configuration ---
set VENV_DIR=venv
set REQUIREMENTS_FILE=systemCore\OpenZephyUI\requirements.txt
set START_SCRIPT=systemCore\OpenZephyUI\backend\start.sh
set RUN_SCRIPT=systemCore\OpenZephyUI\run.sh

REM --- Functions ---

REM Check if Python 3 is available
:check_python3
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Error: Python 3 could not be found. Please install Python 3.
  exit /b 1
)
goto :eof

REM Check if virtual environment exists
:check_venv
if not exist "%VENV_DIR%\Scripts\activate" (
  echo Virtual environment not found. Creating it now...
  python -m venv "%VENV_DIR%"
  if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to create virtual environment.
    exit /b 1
  )
)
goto :eof

REM Activate the virtual environment
:activate_venv
call "%VENV_DIR%\Scripts\activate"
goto :eof

REM Install requirements
:install_requirements
echo Installing requirements...
python -m pip install --upgrade pip
pip install -r "%REQUIREMENTS_FILE%"
if %ERRORLEVEL% NEQ 0 (
  echo Error: Failed to install requirements.
  exit /b 1
)
goto :eof

REM --- Main Script ---

REM Check for Python 3
call :check_python3

REM Check and create virtual environment if needed
call :check_venv

REM Activate the virtual environment
call :activate_venv

REM Install requirements
call :install_requirements

REM Start the backend
echo Starting backend...
bash "%START_SCRIPT%"

REM Run the main script
echo Running application...
bash "%RUN_SCRIPT%"

REM Deactivate the virtual environment (optional)
REM deactivate

echo Application finished.