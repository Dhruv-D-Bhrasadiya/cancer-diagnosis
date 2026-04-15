# Cancer Diagnosis Prediction System - Flask Frontend

A modern, user-friendly web interface for the Cancer Diagnosis Prediction System built with Flask and Bootstrap 5.

## Features

### 🎯 Core Functionality
- **Single Prediction**: Enter patient data as JSON and get instant predictions
- **Batch Prediction**: Upload CSV files for bulk predictions
- **Multiple Models**: Support for 8 different machine learning models
- **Model Comparison**: Compare performance across different models

### 📊 Advanced Features
- **Feature Importance**: Visualize which features matter most for each model
- **Probability Estimates**: Get confidence scores for predictions
- **Decision Scores**: View model decision boundaries
- **Results Export**: Download batch prediction results as CSV

### 🖥️ User Interface
- Clean, modern design with Bootstrap 5
- Responsive layout (works on desktop and mobile)
- Real-time model statistics
- Interactive feature importance charts

## Installation

### 1. Install Dependencies

```bash
# Navigate to project root
cd c:\Users\ADMIN\Desktop\Work\github\cancer-diagnosis

# Activate virtual environment (if using one)
.\.venv\Scripts\Activate.ps1

# Install/update required packages
pip install -r requirements.txt

# Install additional frontend dependencies if needed
pip install flask-cors lime shap
```

### 2. Ensure Models are Trained

The application expects trained models in `outputs/models/` directory:

```
outputs/
└── models/
    ├── logreg_20260414_211340.joblib
    ├── random_forest_20260414_211354.joblib
    └── ... (other model files)
```

If models don't exist, train them first:

```bash
cd src
python main.py
```

### 3. Prepare Data

The frontend loads preprocessing data from `data/processed/`:

```
data/
└── processed/
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── train_test.csv
```

## Running the Application

### Start the Flask Server

```bash
# Navigate to app directory
cd app

# Run the Flask app
python app.py
```

The application will start at `http://localhost:5000`

### For Production Deployment

```bash
# Using Gunicorn (recommended)
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Usage

### 1. Single Prediction

1. Navigate to the home page
2. Select a model from the dropdown
3. Enter input data as JSON in the text area:

```json
{
  "feature_1": 1.5,
  "feature_2": 2.3,
  "feature_3": 0.8,
  "feature_4": -0.2
}
```

4. Click "Predict" to get results
5. View prediction, confidence scores, and feature importance

### 2. Batch Prediction

1. Prepare a CSV file with features as columns
2. Select a model
3. Click "Upload & Predict"
4. View results preview
5. Download full results as CSV

### 3. Model Analysis

- Select a model to see its capabilities
- View feature importance chart
- Compare models by switching between them

## API Endpoints

### Model Management
```
GET  /api/models              - List available models
GET  /api/model-stats         - Get statistics for all models
GET  /api/model-info/<model>  - Get information about a specific model
```

### Predictions
```
POST /api/predict             - Single prediction
POST /api/upload-predict      - Batch predictions from CSV file
```

### Analysis
```
GET  /api/feature-importance/<model>  - Get top 20 features
```

### File Downloads
```
GET  /download/<filename>     - Download prediction results
```

## File Structure

```
app/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── templates/                # HTML templates
│   ├── base.html            # Base template
│   ├── index.html           # Home page
│   ├── about.html           # About page
│   ├── 404.html             # 404 error page
│   └── 500.html             # 500 error page
├── static/                   # Static files
│   ├── css/
│   │   └── style.css        # Custom CSS
│   └── js/
│       ├── main.js          # Main JavaScript functions
│       └── index.js         # Index page specific functions
└── uploads/                  # Directory for uploaded files (created on first use)
```

## Configuration

### Upload Settings

Edit `app.py` to modify:

```python
# Max file size (default: 16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'txt'}
```

### Server Settings

```python
# Development
app.run(debug=True, host='0.0.0.0', port=5000)

# Production (use Gunicorn instead)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Supported ML Models

1. **Logistic Regression** - Fast, interpretable baseline
2. **Random Forest** - Ensemble with feature importance
3. **Gradient Boosting** - Sequential ensemble learning
4. **XGBoost** - Optimized gradient boosting
5. **Ridge Classifier** - Linear with regularization
6. **Support Vector Machine** - Kernel-based classification
7. **K-Nearest Neighbors** - Instance-based learning
8. **Naive Bayes** - Probabilistic classifier

## Input Data Format

### JSON Input (Single Prediction)

```json
{
  "genetic_feature_1": 1.5,
  "genetic_feature_2": -0.8,
  "clinical_feature_1": 2.3,
  "text_feature_tfidf_1": 0.5
}
```

### CSV Input (Batch Prediction)

```csv
genetic_feature_1,genetic_feature_2,clinical_feature_1,text_feature_tfidf_1
1.5,-0.8,2.3,0.5
2.1,-0.5,1.8,0.3
0.8,-1.2,3.0,0.7
```

## Output Formats

### Single Prediction Response

```json
{
  "model": "logreg",
  "predictions": [1],
  "probabilities": [[0.15, 0.25, 0.3, 0.2, 0.1]],
  "decision_scores": [-0.5],
  "input_shape": [1, 50],
  "success": true
}
```

### Batch Prediction Output

CSV file with original features + predictions + probabilities:

```
genetic_feature_1,genetic_feature_2,...,prediction,prob_class_0,prob_class_1,...
1.5,-0.8,...,2,0.1,0.25,...
2.1,-0.5,...,1,0.3,0.2,...
```

## Troubleshooting

### Models Not Loading

```bash
# Check outputs/models directory exists
ls outputs/models/

# Verify model files are .joblib format
file outputs/models/*.joblib
```

### Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Update specific packages
pip install --upgrade flask sklearn pandas numpy
```

### Port Already in Use

```bash
# Change port in app.py
app.run(port=5001)

# Or kill process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### File Upload Issues

```bash
# Ensure uploads directory exists
mkdir app/uploads

# Check file permissions
dir app/uploads

# Verify max upload size in app.py
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

## Performance Tips

1. **Model Selection**: Use LogisticRegression for fast predictions
2. **Batch Size**: Process CSV files in chunks of 1000 rows
3. **Caching**: Models are loaded once at startup
4. **Memory**: Monitor memory usage for large batch predictions

## Security Considerations

- File uploads are validated (extension check)
- Filenames are sanitized using `secure_filename()`
- Input validation for JSON and CSV
- Error messages don't expose system details in production

## Development

### Adding a New Template

1. Create HTML file in `templates/`
2. Extend `base.html`
3. Add route in `app.py`

### Adding New API Endpoint

1. Create function in `app.py`
2. Decorate with `@app.route()`
3. Return JSON response

### Customizing Styling

Edit `static/css/style.css` to modify:
- Colors and fonts
- Layout and spacing
- Responsive breakpoints

## Common Tasks

### Reset Predictions Cache

```bash
rm -r app/uploads/*
```

### View Logs

```bash
# Enable Flask debug mode
export FLASK_ENV=development
export FLASK_DEBUG=1
flask run
```

### Access Model Information

```python
from app import loaded_models
model = loaded_models['logreg']
print(model.get_params())
```

## License

This project is part of the Cancer Diagnosis Prediction System research initiative.

## Support & Documentation

- **Main Project**: See `../README.md`
- **Data Documentation**: See `../data/Instructions.md`
- **Model Evaluation**: See `../src/evaluation/`
- **Feature Engineering**: See `../src/feature/`

## Disclaimer

⚠️ **Important**: This system is for research and educational purposes only. It should not be used for clinical decision-making without professional medical evaluation and validation by qualified healthcare professionals.
