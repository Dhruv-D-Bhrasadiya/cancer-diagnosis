@echo off
REM Flask Application Startup Script for Cancer Diagnosis System

echo.
echo ========================================
echo Cancer Diagnosis Flask Application
echo ========================================
echo.

REM Check if virtual environment is activated
python -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)"
if %ERRORLEVEL% neq 0 (
    echo Virtual environment not activated!
    echo Activating virtual environment...
    call .\.venv\Scripts\activate.bat
)

REM Set Flask environment
set FLASK_APP=app.py
set FLASK_ENV=development

echo.
echo [INFO] Flask Environment: %FLASK_ENV%
echo [INFO] Flask App: %FLASK_APP%
echo.

REM Check if models directory exists
if not exist "..\outputs\models" (
    echo [WARNING] Models directory not found at ..\outputs\models
    echo [INFO] Training models before starting app...
    cd ..\src
    python main.py
    cd ..\app
)

echo.
echo [INFO] Starting Flask development server...
echo [INFO] Open your browser to: http://localhost:5000
echo [INFO] Press CTRL+C to stop the server
echo.

python app.py
