/* Main JavaScript Functions */

// Helper function to show loading state
function setLoading(elementId, isLoading) {
    const element = document.getElementById(elementId);
    if (isLoading) {
        element.disabled = true;
        element.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
    } else {
        element.disabled = false;
        element.innerHTML = element.dataset.originalText || 'Submit';
    }
}

// Helper function to show error
function showError(message, containerId = null) {
    const alertHtml = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            <i class="bi bi-exclamation-triangle-fill"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    if (containerId) {
        document.getElementById(containerId).insertAdjacentHTML('beforebegin', alertHtml);
    } else {
        document.querySelector('main').insertAdjacentHTML('afterbegin', alertHtml);
    }
}

// Helper function to show success
function showSuccess(message, containerId = null) {
    const alertHtml = `
        <div class="alert alert-success alert-dismissible fade show" role="alert">
            <i class="bi bi-check-circle-fill"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    if (containerId) {
        document.getElementById(containerId).insertAdjacentHTML('beforebegin', alertHtml);
    } else {
        document.querySelector('main').insertAdjacentHTML('afterbegin', alertHtml);
    }
}

// API call helper
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(endpoint, options);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP Error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Format number to 4 decimal places
function formatNumber(num) {
    return parseFloat(num).toFixed(4);
}

// Create HTML for probabilities
function createProbabilitiesHTML(probabilities, predictedClass) {
    if (!probabilities || !Array.isArray(probabilities[0])) {
        return '';
    }

    const probs = probabilities[0];
    let html = '<div class="probability-items">';

    probs.forEach((prob, index) => {
        const classNum = index + 1;   // model classes are 1-indexed
        const percentage = (prob * 100).toFixed(2);
        const ispredicted = classNum === predictedClass;
        const barColor = ispredicted ? '#0d6efd' : '#6c757d';
        const bold = ispredicted ? 'fw-bold text-primary' : '';
        html += `
            <div class="probability-item ${bold}">
                <span class="me-2" style="min-width:70px; display:inline-block;">
                    <strong>Class ${classNum}${ispredicted ? ' ✓' : ''}:</strong>
                </span>
                <div class="probability-bar" style="display:inline-block; width:55%; background:#e9ecef; border-radius:4px; height:14px; vertical-align:middle;">
                    <div style="width:${percentage}%; background:${barColor}; height:14px; border-radius:4px;"></div>
                </div>
                <span class="ms-2">${percentage}%</span>
            </div>
        `;
    });

    html += '</div>';
    return html;
}

// Clear results
function clearResults() {
    document.getElementById('predictionResult').classList.add('d-none');
    document.getElementById('batchResult').classList.add('d-none');
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Validate JSON
function isValidJSON(str) {
    try {
        JSON.parse(str);
        return true;
    } catch (e) {
        return false;
    }
}
