# Sample Data for Testing Cancer Diagnosis Frontend

This directory contains sample data and scripts for testing the Flask frontend without needing to train full models first.

## Sample Input Format

### JSON for Single Prediction

```json
{
  "gene_class": 1.5,
  "variant_class": 2.3,
  "clinical_feature_1": 0.8,
  "text_feature_tfidf_1": -0.2,
  "text_feature_tfidf_2": 0.5
}
```

### CSV for Batch Prediction

Create a file `sample_input.csv`:

```csv
gene_class,variant_class,clinical_feature_1,text_feature_tfidf_1,text_feature_tfidf_2
1.5,2.3,0.8,-0.2,0.5
2.1,-0.5,1.8,0.3,0.2
0.8,-1.2,3.0,0.7,-0.1
```

## Generating Sample Data

To generate sample data for testing:

```bash
python generate_sample_data.py
```

This will create:
- `sample_predictions.csv` - 100 sample records
- `sample_results.csv` - Prediction results

## Using with Flask Frontend

### For Single Predictions

1. Open http://localhost:5000
2. Select a model
3. Paste the JSON input in the text area
4. Click "Predict"

### For Batch Predictions

1. Open http://localhost:5000
2. Upload `sample_input.csv`
3. Select a model
4. Click "Upload & Predict"
5. Download results

## Expected Model Output

Models should return:
- **Prediction**: Class label (integer)
- **Probabilities**: Array of probabilities for each class
- **Confidence**: Highest probability value

Example output structure:

```python
{
    'predictions': [1, 2, 0],  # Predicted classes
    'probabilities': [
        [0.1, 0.8, 0.1],  # Sample 1
        [0.2, 0.3, 0.5],  # Sample 2
        [0.7, 0.2, 0.1]   # Sample 3
    ]
}
```

## Cancer Type Mapping

Assuming the model has been trained on cancer types:

- Class 0: Type A
- Class 1: Type B
- Class 2: Type C
- ... (add more based on your dataset)

Update this mapping in `app.py` if needed:

```python
CLASS_NAMES = {
    0: 'Cancer Type A',
    1: 'Cancer Type B',
    2: 'Cancer Type C',
}
```

## Troubleshooting

### Model Not Found

```bash
# Ensure models are in the correct location
ls outputs/models/

# Check model file format
file outputs/models/*.joblib
```

### Invalid Input Format

- Ensure JSON is valid (use JSONLint)
- CSV must have header row
- Column names must match training features

### No Predictions Returned

1. Check Flask logs for errors
2. Verify model file is not corrupted
3. Ensure input dimensions match training data

## Next Steps

1. Train actual models using `src/main.py`
2. Save models to `outputs/models/`
3. Run Flask app: `python app.py`
4. Test with sample data
5. Integrate with real patient data
