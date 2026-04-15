"""
Flask application for cancer diagnosis prediction.
"""

import os
import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from feature.preprocessing import preprocess_pipeline
from data.loader import load_processed_data
from evaluation.metrics import evaluate_classification
from evaluation.interpretability import get_feature_importance

# Flask App Configuration
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state for loaded models and preprocessor
loaded_models = {}
preprocessor_info = {
    'vectorizers': None,
    'X_train': None,
    'y_train': None,
    'feature_names': None
}

ALLOWED_EXTENSIONS = {'csv', 'txt'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load all trained models from outputs/models directory."""
    global loaded_models
    
    models_dir = project_root / "outputs" / "models"
    
    if not models_dir.exists():
        print(f"[WARNING] Models directory not found: {models_dir}")
        return False
    
    for model_file in models_dir.glob("*.joblib"):
        try:
            model_name = model_file.stem
            model = joblib.load(str(model_file))
            loaded_models[model_name] = model
            print(f"[INFO] Loaded model: {model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_file}: {e}")
    
    return len(loaded_models) > 0


def load_preprocessing_data():
    """Load preprocessing data and vectorizers."""
    global preprocessor_info
    
    try:
        # Load processed data
        train_df, val_df, test_df, test_comp_df = load_processed_data()
        
        # Extract features from training data for preprocessing reference
        from feature.preprocessing import get_all_preprocessing_configs
        configs = get_all_preprocessing_configs()
        
        # Use first config as default
        if configs:
            config = configs[0]
            X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(
                train_df, val_df, test_df, config=config
            )
            
            preprocessor_info['X_train'] = X_train
            preprocessor_info['y_train'] = y_train
            
            # Try to extract feature names if available
            if hasattr(X_train, 'get_feature_names_out'):
                preprocessor_info['feature_names'] = X_train.get_feature_names_out()
            elif hasattr(X_train, 'columns'):
                preprocessor_info['feature_names'] = X_train.columns.tolist()
            
            print("[INFO] Preprocessing data loaded successfully")
            return True
    except Exception as e:
        print(f"[WARNING] Could not load preprocessing data: {e}")
        return False


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html', models=list(loaded_models.keys()))


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')


@app.route('/api/models')
def get_models():
    """Get list of available models."""
    return jsonify({
        'models': list(loaded_models.keys()),
        'count': len(loaded_models)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction with the selected model."""
    try:
        data = request.get_json()
        
        if not data or 'model' not in data or 'input' not in data:
            return jsonify({'error': 'Missing required fields: model, input'}), 400
        
        model_name = data['model']
        input_data = data['input']
        
        # Validate model exists
        if model_name not in loaded_models:
            return jsonify({'error': f'Model not found: {model_name}'}), 404
        
        model = loaded_models[model_name]
        
        # Convert input to proper format
        if isinstance(input_data, dict):
            # Single sample input
            X = np.array([list(input_data.values())])
        elif isinstance(input_data, list):
            X = np.array(input_data)
        else:
            return jsonify({'error': 'Invalid input format'}), 400
        
        # Make predictions
        predictions = model.predict(X)
        
        # Get probabilities if available
        probabilities = None
        try:
            probs = model.predict_proba(X)
            probabilities = probs.tolist()
        except AttributeError:
            pass
        
        # Get decision scores if available
        decision_scores = None
        try:
            scores = model.decision_function(X)
            decision_scores = scores.tolist()
        except (AttributeError, NotImplementedError):
            pass
        
        response = {
            'model': model_name,
            'predictions': predictions.tolist(),
            'probabilities': probabilities,
            'decision_scores': decision_scores,
            'input_shape': X.shape,
            'success': True
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/upload-predict', methods=['POST'])
def upload_predict():
    """Upload CSV and make predictions."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        if 'model' not in request.form:
            return jsonify({'error': 'No model selected'}), 400
        
        file = request.files['file']
        model_name = request.form['model']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Validate model exists
        if model_name not in loaded_models:
            return jsonify({'error': f'Model not found: {model_name}'}), 404
        
        model = loaded_models[model_name]
        
        # Save and load CSV
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Get probabilities if available
        probabilities = None
        try:
            probs = model.predict_proba(df)
            probabilities = probs.tolist()
        except AttributeError:
            pass
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        
        if probabilities is not None:
            for i, prob in enumerate(probabilities[0]):
                df[f'prob_class_{i}'] = [p[i] for p in probabilities]
        
        # Save results
        results_filename = f"predictions_{secure_filename(filename)}"
        results_path = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
        df.to_csv(results_path, index=False)
        
        response = {
            'model': model_name,
            'rows_processed': len(df),
            'predictions': predictions.tolist()[:100],  # Return first 100 for preview
            'total_predictions': len(predictions),
            'results_file': results_filename,
            'success': True
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"[ERROR] Upload prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/feature-importance/<model_name>')
def get_feature_importance_api(model_name):
    """Get feature importance for a model."""
    try:
        if model_name not in loaded_models:
            return jsonify({'error': f'Model not found: {model_name}'}), 404
        
        model = loaded_models[model_name]
        
        # Get feature importance
        try:
            importance_list = get_feature_importance(
                model,
                feature_names=preprocessor_info['feature_names'],
                top_k=20
            )
            
            features = [item['feature'] for item in importance_list]
            scores = [item['importance'] for item in importance_list]
            
            return jsonify({
                'model': model_name,
                'features': features,
                'scores': scores,
                'success': True
            })
        except Exception as e:
            return jsonify({'error': f'Model does not support feature importance: {str(e)}'}), 400
    
    except Exception as e:
        print(f"[ERROR] Feature importance error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/model-info/<model_name>')
def model_info(model_name):
    """Get information about a specific model."""
    try:
        if model_name not in loaded_models:
            return jsonify({'error': f'Model not found: {model_name}'}), 404
        
        model = loaded_models[model_name]
        
        info = {
            'name': model_name,
            'type': type(model).__name__,
            'has_predict_proba': hasattr(model, 'predict_proba'),
            'has_decision_function': hasattr(model, 'decision_function'),
            'has_coef': hasattr(model, 'coef_'),
            'has_feature_importance': hasattr(model, 'feature_importances_'),
            'parameters': str(model.get_params()) if hasattr(model, 'get_params') else 'N/A'
        }
        
        return jsonify(info)
    
    except Exception as e:
        print(f"[ERROR] Model info error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/model-stats')
def model_stats():
    """Get statistics about all loaded models."""
    try:
        stats = []
        
        for model_name, model in loaded_models.items():
            stat = {
                'name': model_name,
                'type': type(model).__name__,
                'has_predict_proba': hasattr(model, 'predict_proba'),
                'has_feature_importance': hasattr(model, 'feature_importances_'),
            }
            stats.append(stat)
        
        return jsonify({
            'total_models': len(loaded_models),
            'models': stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download prediction results."""
    try:
        filename = secure_filename(filename)
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500


if __name__ == '__main__':
    print("[INFO] Starting Cancer Diagnosis Flask App...")
    print("[INFO] Loading trained models...")
    
    if load_models():
        print(f"[INFO] Successfully loaded {len(loaded_models)} models")
    else:
        print("[WARNING] No models loaded. The app will still run but predictions won't be available.")
    
    print("[INFO] Loading preprocessing data...")
    load_preprocessing_data()
    
    print("[INFO] Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
