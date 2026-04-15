/* Index Page JavaScript */

let featureImportanceChart = null;

document.addEventListener('DOMContentLoaded', function() {
    loadModels();
    loadModelStats();
    
    // Event Listeners
    document.getElementById('modelSelect').addEventListener('change', function() {
        const modelName = this.value;
        if (modelName) {
            loadModelInfo(modelName);
            loadFeatureImportance(modelName);
        } else {
            document.getElementById('modelInfo').classList.add('d-none');
            clearFeatureImportanceChart();
        }
    });
    
    document.getElementById('singlePredictionForm').addEventListener('submit', handleSinglePrediction);
    document.getElementById('batchPredictionForm').addEventListener('submit', handleBatchPrediction);
});

// Load available models
async function loadModels() {
    try {
        const data = await apiCall('/api/models');
        const select = document.getElementById('modelSelect');
        const batchSelect = document.getElementById('batchModelSelect');
        
        // Clear existing options (except the default)
        select.innerHTML = '<option value="">-- Choose a model --</option>';
        batchSelect.innerHTML = '<option value="">-- Choose a model --</option>';
        
        // Add model options
        data.models.forEach(model => {
            const option1 = document.createElement('option');
            option1.value = model;
            option1.textContent = model;
            select.appendChild(option1);
            
            const option2 = document.createElement('option');
            option2.value = model;
            option2.textContent = model;
            batchSelect.appendChild(option2);
        });
    } catch (error) {
        console.error('Error loading models:', error);
        showError('Failed to load models: ' + error.message);
    }
}

// Load model statistics
async function loadModelStats() {
    try {
        const data = await apiCall('/api/model-stats');
        document.getElementById('totalModels').textContent = data.total_models;
    } catch (error) {
        console.error('Error loading model stats:', error);
    }
}

// Load and display model information
async function loadModelInfo(modelName) {
    try {
        const data = await apiCall(`/api/model-info/${modelName}`);
        
        let infoContent = `
            <strong>Model Type:</strong> ${data.type}<br>
            <strong>Probability Support:</strong> ${data.has_predict_proba ? '✓' : '✗'}<br>
            <strong>Feature Importance:</strong> ${data.has_feature_importance ? '✓' : '✗'}
        `;
        
        document.getElementById('modelInfoContent').innerHTML = infoContent;
        document.getElementById('modelInfo').classList.remove('d-none');
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Load and display feature importance
async function loadFeatureImportance(modelName) {
    try {
        const data = await apiCall(`/api/feature-importance/${modelName}`);
        
        if (data.features && data.features.length > 0) {
            displayFeatureImportanceChart(data.features, data.scores, modelName);
            document.getElementById('featureImportanceContainer').classList.remove('d-none');
            document.getElementById('noFeatureImportance').classList.add('d-none');
        } else {
            clearFeatureImportanceChart();
            document.getElementById('featureImportanceContainer').classList.add('d-none');
            document.getElementById('noFeatureImportance').classList.remove('d-none');
        }
    } catch (error) {
        console.error('Error loading feature importance:', error);
        clearFeatureImportanceChart();
        document.getElementById('featureImportanceContainer').classList.add('d-none');
        document.getElementById('noFeatureImportance').classList.remove('d-none');
    }
}

// Display feature importance chart
function displayFeatureImportanceChart(features, scores, modelName) {
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }
    
    featureImportanceChart = new Chart(ctx, {
        type: 'barh',
        data: {
            labels: features,
            datasets: [{
                label: 'Importance Score',
                data: scores,
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: `Feature Importance - ${modelName}`
                }
            },
            scales: {
                x: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Clear feature importance chart
function clearFeatureImportanceChart() {
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
        featureImportanceChart = null;
    }
}

// Handle single prediction
async function handleSinglePrediction(e) {
    e.preventDefault();
    
    const modelSelect = document.getElementById('modelSelect');
    const inputData = document.getElementById('inputData');
    
    if (!modelSelect.value) {
        showError('Please select a model');
        return;
    }
    
    if (!inputData.value.trim()) {
        showError('Please enter input data');
        return;
    }
    
    if (!isValidJSON(inputData.value)) {
        showError('Invalid JSON format. Please check your input.');
        return;
    }
    
    try {
        const btn = e.target.querySelector('button[type="submit"]');
        btn.dataset.originalText = btn.innerHTML;
        setLoading(btn.id || null, true);
        
        const data = await apiCall('/api/predict', 'POST', {
            model: modelSelect.value,
            input: JSON.parse(inputData.value)
        });
        
        displaySinglePredictionResult(data);
        showSuccess('Prediction completed successfully!');
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Prediction failed: ' + error.message);
    } finally {
        const btn = document.querySelector('#singlePredictionForm button[type="submit"]');
        btn.disabled = false;
        btn.innerHTML = btn.dataset.originalText || '<i class="bi bi-play-circle"></i> Predict';
    }
}

// Display single prediction result
function displaySinglePredictionResult(data) {
    const resultDiv = document.getElementById('predictionResult');
    const predictionValue = document.getElementById('predictionValue');
    const probabilitiesDiv = document.getElementById('probabilitiesDiv');
    const probabilitiesContent = document.getElementById('probabilitiesContent');
    
    predictionValue.textContent = data.predictions[0];
    
    if (data.probabilities) {
        probabilitiesContent.innerHTML = createProbabilitiesHTML(data.probabilities);
        probabilitiesDiv.classList.remove('d-none');
    } else {
        probabilitiesDiv.classList.add('d-none');
    }
    
    resultDiv.classList.remove('d-none');
}

// Handle batch prediction
async function handleBatchPrediction(e) {
    e.preventDefault();
    
    const form = e.target;
    const fileInput = document.getElementById('csvFile');
    const modelSelect = document.getElementById('batchModelSelect');
    
    if (!modelSelect.value) {
        showError('Please select a model');
        return;
    }
    
    if (!fileInput.files.length) {
        showError('Please select a file');
        return;
    }
    
    try {
        const btn = form.querySelector('button[type="submit"]');
        btn.dataset.originalText = btn.innerHTML;
        setLoading(btn.id || null, true);
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('model', modelSelect.value);
        
        const response = await fetch('/api/upload-predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Upload failed');
        }
        
        const data = await response.json();
        displayBatchPredictionResult(data);
        showSuccess(`Batch prediction completed! Processed ${data.rows_processed} rows.`);
        
    } catch (error) {
        console.error('Batch prediction error:', error);
        showError('Batch prediction failed: ' + error.message);
    } finally {
        const btn = form.querySelector('button[type="submit"]');
        btn.disabled = false;
        btn.innerHTML = btn.dataset.originalText || '<i class="bi bi-upload"></i> Upload & Predict';
    }
}

// Display batch prediction result
function displayBatchPredictionResult(data) {
    const resultDiv = document.getElementById('batchResult');
    document.getElementById('rowsProcessed').textContent = data.rows_processed;
    document.getElementById('totalPredictions').textContent = data.total_predictions;
    
    const downloadLink = document.getElementById('downloadResults');
    downloadLink.href = `/download/${data.results_file}`;
    
    resultDiv.classList.remove('d-none');
}
