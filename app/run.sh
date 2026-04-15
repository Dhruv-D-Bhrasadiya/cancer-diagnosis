#!/bin/bash

# Flask Application Startup Script for Cancer Diagnosis System

echo ""
echo "========================================"
echo "Cancer Diagnosis Flask Application"
echo "========================================"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not activated!"
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Set Flask environment
export FLASK_APP=app.py
export FLASK_ENV=development

echo ""
echo "[INFO] Flask Environment: $FLASK_ENV"
echo "[INFO] Flask App: $FLASK_APP"
echo ""

# Check if models directory exists
if [ ! -d "../outputs/models" ]; then
    echo "[WARNING] Models directory not found at ../outputs/models"
    echo "[INFO] Training models before starting app..."
    cd ../src
    python main.py
    cd ../app
fi

echo ""
echo "[INFO] Starting Flask development server..."
echo "[INFO] Open your browser to: http://localhost:5000"
echo "[INFO] Press CTRL+C to stop the server"
echo ""

python app.py
