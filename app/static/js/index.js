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
    const gene      = document.getElementById('geneInput').value.trim();
    const variation = document.getElementById('variationInput').value.trim();
    const text      = document.getElementById('textInput').value.trim();

    if (!modelSelect.value) { showError('Please select a model'); return; }
    if (!gene || !variation) { showError('Please enter Gene and Variation'); return; }

    try {
        const btn = e.target.querySelector('button[type="submit"]');
        btn.dataset.originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting...';

        const data = await apiCall('/api/predict', 'POST', {
            model: modelSelect.value,
            gene, variation, text
        });

        displaySinglePredictionResult(data);
        showSuccess('Prediction completed successfully!');

        // Pass X_encoded (the actual feature vector) to LIME/SHAP
        enableXaiButtons(modelSelect.value, data.X_encoded);

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

    const predictedClass = data.predictions[0];
    predictionValue.textContent = predictedClass;

    if (data.probabilities) {
        probabilitiesContent.innerHTML = createProbabilitiesHTML(data.probabilities, predictedClass);
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


// ═══════════════════════════════════════════════════════════════════════════
// XAI — LIME & SHAP
// ═══════════════════════════════════════════════════════════════════════════

let limeChart      = null;
let shapGlobalChart = null;

// Stores the last prediction payload so LIME/SHAP use the same input
let lastXaiPayload = null;

// ── Enable buttons after a successful prediction ─────────────────────────
function enableXaiButtons(modelName, inputObj) {
    lastXaiPayload = { model: modelName, input: inputObj };

    document.getElementById('runLimeBtn').disabled = false;
    document.getElementById('runShapBtn').disabled = false;
    document.getElementById('xaiNote').textContent =
        'Ready — click Run LIME or Run SHAP to explain this prediction.';

    // Reset previous results
    document.getElementById('limeSection').classList.add('d-none');
    document.getElementById('shapSection').classList.add('d-none');
    document.getElementById('xaiError').classList.add('d-none');
}

// ── LIME ─────────────────────────────────────────────────────────────────

document.getElementById('runLimeBtn').addEventListener('click', async () => {
    if (!lastXaiPayload) return;

    showXaiLoading('lime');
    hideXaiError();

    try {
        const result = await apiCall('/api/explain/lime', 'POST', lastXaiPayload);
        renderLime(result);
    } catch (err) {
        showXaiError('LIME failed: ' + err.message);
    } finally {
        hideXaiLoading('lime');
    }
});

function renderLime(data) {
    // Badge showing predicted class
    document.getElementById('limePredClassBadge').textContent = `Class ${data.predicted_class}`;

    // Populate class selector dropdown
    const sel = document.getElementById('limeClassSelect');
    sel.innerHTML = '';
    data.all_classes.forEach(cls => {
        const opt = document.createElement('option');
        opt.value = cls.class_index;
        opt.textContent = cls.class_name;
        // Pre-select the predicted class
        if (cls.class_index === data.predicted_class) opt.selected = true;
        sel.appendChild(opt);
    });

    // Draw bar chart for predicted class
    const predEntry = data.all_classes.find(c => c.class_index === data.predicted_class)
        || data.all_classes[0];
    drawLimeChart(predEntry);
    renderLimeTable(predEntry);

    // Update chart + table when user changes class
    sel.onchange = () => {
        const chosen = data.all_classes.find(c => String(c.class_index) === sel.value);
        if (chosen) { drawLimeChart(chosen); renderLimeTable(chosen); }
    };

    document.getElementById('limeSection').classList.remove('d-none');
}

function drawLimeChart(classEntry) {
    const ctx = document.getElementById('limeChart').getContext('2d');
    if (limeChart) limeChart.destroy();

    const labels  = classEntry.features.map(f => truncLabel(f.feature, 32));
    const weights = classEntry.features.map(f => f.weight);
    // Blue for positive (supports class), red for negative (opposes class)
    const bgColors = weights.map(w => w >= 0 ? 'rgba(13,110,253,0.7)' : 'rgba(220,53,69,0.7)');
    const bdColors = weights.map(w => w >= 0 ? 'rgba(13,110,253,1)'   : 'rgba(220,53,69,1)');

    limeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: `LIME weights — ${classEntry.class_name}`,
                data: weights,
                backgroundColor: bgColors,
                borderColor: bdColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: c => ` ${c.parsed.x.toFixed(4)}` } }
            },
            scales: { x: { title: { display: true, text: 'Weight' } } }
        }
    });
}

function renderLimeTable(classEntry) {
    const tbody = document.getElementById('limeTableBody');
    tbody.innerHTML = '';
    classEntry.features.forEach(f => {
        const dir = f.weight >= 0
            ? '<span class="badge bg-primary">▲ supports</span>'
            : '<span class="badge bg-danger">▼ opposes</span>';
        tbody.insertAdjacentHTML('beforeend', `
            <tr>
                <td class="small text-break">${escHtml(f.feature)}</td>
                <td class="text-end small">${f.weight.toFixed(4)}</td>
                <td>${dir}</td>
            </tr>`);
    });
}

// ── SHAP ─────────────────────────────────────────────────────────────────

document.getElementById('runShapBtn').addEventListener('click', async () => {
    if (!lastXaiPayload) return;

    showXaiLoading('shap');
    hideXaiError();

    try {
        const result = await apiCall('/api/explain/shap', 'POST', lastXaiPayload);
        renderShap(result);
    } catch (err) {
        showXaiError('SHAP failed: ' + err.message);
    } finally {
        hideXaiLoading('shap');
    }
});

function renderShap(data) {
    // Global importance bar chart
    drawShapGlobalChart(data.top_features);

    // Populate class selector
    const sel = document.getElementById('shapClassSelect');
    sel.innerHTML = '';
    data.shap_values.forEach(cls => {
        const opt = document.createElement('option');
        opt.value = cls.class_index;
        opt.textContent = cls.class_name;
        if (cls.class_index === data.predicted_class) opt.selected = true;
        sel.appendChild(opt);
    });

    // Table for predicted class
    const predEntry = data.shap_values.find(c => c.class_index === data.predicted_class)
        || data.shap_values[0];
    renderShapTable(predEntry);

    sel.onchange = () => {
        const chosen = data.shap_values.find(c => String(c.class_index) === sel.value);
        if (chosen) renderShapTable(chosen);
    };

    // Expected values
    if (data.expected_value && data.expected_value.length) {
        document.getElementById('shapExpectedValueContent').textContent =
            ' ' + data.expected_value.map(v => v.toFixed(4)).join(', ');
        document.getElementById('shapExpectedValue').classList.remove('d-none');
    }

    document.getElementById('shapSection').classList.remove('d-none');
}

function drawShapGlobalChart(topFeatures) {
    const ctx = document.getElementById('shapGlobalChart').getContext('2d');
    if (shapGlobalChart) shapGlobalChart.destroy();

    const labels = topFeatures.map(f => truncLabel(f.feature, 32));
    const values = topFeatures.map(f => f.mean_abs_shap);

    shapGlobalChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Mean |SHAP value|',
                data: values,
                backgroundColor: 'rgba(111,66,193,0.65)',
                borderColor: 'rgba(111,66,193,1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: c => ` ${c.parsed.x.toFixed(4)}` } }
            },
            scales: {
                x: { beginAtZero: true, title: { display: true, text: 'Mean |SHAP value|' } }
            }
        }
    });
}

function renderShapTable(classEntry) {
    const tbody = document.getElementById('shapTableBody');
    tbody.innerHTML = '';
    classEntry.features.forEach(f => {
        const dir = f.shap_value >= 0
            ? '<span class="badge bg-success">▲ increases prob</span>'
            : '<span class="badge bg-danger">▼ decreases prob</span>';
        tbody.insertAdjacentHTML('beforeend', `
            <tr>
                <td class="small text-break">${escHtml(f.feature)}</td>
                <td class="text-end small">${f.shap_value.toFixed(4)}</td>
                <td>${dir}</td>
            </tr>`);
    });
}

// ── Shared helpers ────────────────────────────────────────────────────────

function showXaiLoading(type) {
    document.getElementById(`${type}Loading`).classList.remove('d-none');
    document.getElementById(`run${type.charAt(0).toUpperCase() + type.slice(1)}Btn`).disabled = true;
}

function hideXaiLoading(type) {
    document.getElementById(`${type}Loading`).classList.add('d-none');
    document.getElementById(`run${type.charAt(0).toUpperCase() + type.slice(1)}Btn`).disabled = false;
}

function showXaiError(msg) {
    const el = document.getElementById('xaiError');
    el.textContent = msg;
    el.classList.remove('d-none');
}

function hideXaiError() {
    document.getElementById('xaiError').classList.add('d-none');
}

function truncLabel(str, max) {
    return str.length > max ? str.slice(0, max - 1) + '…' : str;
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
