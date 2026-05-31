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
from scipy.sparse import hstack, issparse
from sklearn.preprocessing import normalize
import re
import traceback

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from evaluation.metrics import evaluate_classification
from evaluation.interpretability import get_feature_importance, explain_with_lime, explain_with_shap

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
        # Skip vectorizer files — only load actual models
        if any(x in model_file.stem for x in ['vectorizer', 'vec']):
            continue
        try:
            model_name = model_file.stem
            model = joblib.load(str(model_file))
            loaded_models[model_name] = model
            print(f"[INFO] Loaded model: {model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_file}: {e}")

    return len(loaded_models) > 0


def load_preprocessing_data():
    """
    Load vectorizers saved by train_model.py and rebuild X_train
    for use as LIME/SHAP background data.
    """
    global preprocessor_info

    try:
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()

    def clean_text(text):
        if not isinstance(text, str):
            return ''
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text).lower()
        return ' '.join(w for w in text.split() if w not in stop_words)

    models_dir = project_root / "outputs" / "models"

    # Check vectorizers saved by train_model.py exist
    gene_vec_path = models_dir / "gene_vectorizer.joblib"
    var_vec_path  = models_dir / "var_vectorizer.joblib"
    text_vec_path = models_dir / "text_vectorizer.joblib"

    if not all(p.exists() for p in [gene_vec_path, var_vec_path, text_vec_path]):
        print("[WARNING] Vectorizers not found in outputs/models/. LIME/SHAP background unavailable.")
        return False

    try:
        gene_vec = joblib.load(gene_vec_path)
        var_vec  = joblib.load(var_vec_path)
        text_vec = joblib.load(text_vec_path)

        preprocessor_info['vectorizers'] = {
            'gene': gene_vec,
            'var':  var_vec,
            'text': text_vec
        }

        # Rebuild X_train from processed CSV for LIME/SHAP background
        train_csv = project_root / "data/processed/train_data.csv"
        if not train_csv.exists():
            print("[WARNING] train_data.csv not found. LIME/SHAP background unavailable.")
            return False

        train_df = pd.read_csv(train_csv)
        train_df['TEXT']      = train_df['TEXT'].fillna('').apply(clean_text)
        train_df['Gene']      = train_df['Gene'].fillna('').str.lower()
        train_df['Variation'] = train_df['Variation'].fillna('').str.lower()

        X_train = hstack([
            gene_vec.transform(train_df['Gene']),
            var_vec.transform(train_df['Variation']),
            normalize(text_vec.transform(train_df['TEXT']), axis=0)
        ]).tocsr()  # CSR supports row indexing needed by LIME/SHAP

        preprocessor_info['X_train']  = X_train
        preprocessor_info['y_train']  = train_df['Class'].values

        # Build feature names: gene vocab + var vocab + text vocab
        feature_names = (
            list(gene_vec.get_feature_names_out()) +
            list(var_vec.get_feature_names_out()) +
            list(text_vec.get_feature_names_out())
        )
        preprocessor_info['feature_names'] = feature_names

        print(f"[INFO] Preprocessing data loaded. X_train shape: {X_train.shape}")
        print(f"[INFO] Feature names: {len(feature_names)} total")
        return True

    except Exception as e:
        print(f"[WARNING] Could not load preprocessing data: {e}")
        traceback.print_exc()
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
    """
    Make prediction with the selected model.
    Accepts either:
      - Clinical input: {"model": "logreg", "gene": "BRCA1", "variation": "R248Q", "text": "..."}
      - Raw feature vector: {"model": "logreg", "input": {"f1": 0.1, ...}}
    """
    try:
        data = request.get_json()

        if not data or 'model' not in data:
            return jsonify({'error': 'Missing required field: model'}), 400

        model_name = data['model']
        if model_name not in loaded_models:
            return jsonify({'error': f'Model not found: {model_name}'}), 404

        model = loaded_models[model_name]

        # ── Clinical input path (Gene + Variation + Text) ────────────────────
        if 'gene' in data or 'variation' in data:
            vecs = preprocessor_info.get('vectorizers')
            if vecs is None:
                return jsonify({'error': 'Vectorizers not loaded.'}), 503

            import re
            from nltk.corpus import stopwords
            try:
                stop_words = set(stopwords.words('english'))
            except Exception:
                stop_words = set()

            def clean_text(text):
                if not isinstance(text, str): return ''
                text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
                text = re.sub(r'\s+', ' ', text).lower()
                return ' '.join(w for w in text.split() if w not in stop_words)

            gene      = str(data.get('gene', '')).lower()
            variation = str(data.get('variation', '')).lower()
            text      = clean_text(data.get('text', ''))

            X = hstack([
                vecs['gene'].transform([gene]),
                vecs['var'].transform([variation]),
                normalize(vecs['text'].transform([text]), axis=0)
            ]).tocsr()

        # ── Raw feature vector path ──────────────────────────────────────────
        elif 'input' in data:
            input_data = data['input']
            if isinstance(input_data, dict):
                X = np.array([list(input_data.values())])
            elif isinstance(input_data, list):
                X = np.array(input_data)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
            else:
                return jsonify({'error': 'Invalid input format'}), 400
        else:
            return jsonify({'error': 'Provide either gene/variation/text or input fields'}), 400

        # ── Predict ──────────────────────────────────────────────────────────
        predictions = model.predict(X)

        probabilities = None
        try:
            probs = model.predict_proba(X)
            probabilities = probs.tolist()
        except AttributeError:
            pass

        decision_scores = None
        try:
            scores = model.decision_function(X)
            decision_scores = scores.tolist()
        except (AttributeError, NotImplementedError):
            pass

        # Store X as list for LIME/SHAP reuse
        X_dense = X.toarray() if issparse(X) else X
        return jsonify({
            'model': model_name,
            'predictions': predictions.tolist(),
            'probabilities': probabilities,
            'decision_scores': decision_scores,
            'input_shape': list(X_dense.shape),
            'X_encoded': X_dense.tolist(),   # returned so JS can pass it to LIME/SHAP
            'success': True
        })

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


@app.route('/api/explain/lime', methods=['POST'])
def explain_lime():
    """
    Run LIME on a single input sample and return feature weights.

    POST body (JSON):
    {
        "model": "<model_name>",
        "input": {"feature1": val, ...}   OR  [[val1, val2, ...]]
    }
    """
    try:
        data = request.get_json()

        if not data or 'model' not in data or 'input' not in data:
            return jsonify({'error': 'Missing required fields: model, input'}), 400

        model_name = data['model']
        input_data = data['input']

        if model_name not in loaded_models:
            return jsonify({'error': f'Model not found: {model_name}'}), 404

        model = loaded_models[model_name]

        if isinstance(input_data, dict):
            X_sample = np.array([list(input_data.values())])
        elif isinstance(input_data, list):
            X_sample = np.array(input_data)
            if X_sample.ndim == 1:
                X_sample = X_sample.reshape(1, -1)
        else:
            return jsonify({'error': 'Invalid input format'}), 400

        X_train = preprocessor_info.get('X_train')
        feature_names = preprocessor_info.get('feature_names')

        if X_train is None:
            return jsonify({'error': 'Training data not loaded. Cannot run LIME without background data.'}), 503

        result = explain_with_lime(
            model=model,
            X_train=X_train,
            X_sample=X_sample,
            feature_names=feature_names,
            num_features=data.get('num_features', 10),
            num_samples=data.get('num_samples', 500)
        )

        return jsonify({'success': True, 'model': model_name, **result})

    except Exception as e:
        print(f"[ERROR] LIME explanation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/explain/shap', methods=['POST'])
def explain_shap():
    """
    Run SHAP on a single input sample and return SHAP values.

    POST body (JSON):
    {
        "model": "<model_name>",
        "input": {"feature1": val, ...}   OR  [[val1, val2, ...]]
    }
    """
    try:
        data = request.get_json()

        if not data or 'model' not in data or 'input' not in data:
            return jsonify({'error': 'Missing required fields: model, input'}), 400

        model_name = data['model']
        input_data = data['input']

        if model_name not in loaded_models:
            return jsonify({'error': f'Model not found: {model_name}'}), 404

        model = loaded_models[model_name]

        if isinstance(input_data, dict):
            X_sample = np.array([list(input_data.values())])
        elif isinstance(input_data, list):
            X_sample = np.array(input_data)
            if X_sample.ndim == 1:
                X_sample = X_sample.reshape(1, -1)
        else:
            return jsonify({'error': 'Invalid input format'}), 400

        X_train = preprocessor_info.get('X_train')
        feature_names = preprocessor_info.get('feature_names')

        if X_train is None:
            return jsonify({'error': 'Training data not loaded. Cannot run SHAP without background data.'}), 503

        result = explain_with_shap(
            model=model,
            X_train=X_train,
            X_sample=X_sample,
            feature_names=feature_names,
            top_k=data.get('top_k', 15),
            background_samples=data.get('background_samples', 50)
        )

        return jsonify({'success': True, 'model': model_name, **result})

    except Exception as e:
        print(f"[ERROR] SHAP explanation error: {e}")
        traceback.print_exc()
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
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
