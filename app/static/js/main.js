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
function createProbabilitiesHTML(probabilities) {
    if (!probabilities || !Array.isArray(probabilities[0])) {
        return '';
    }
    
    const probs = probabilities[0];
    let html = '<div class="probability-items">';
    
    probs.forEach((prob, index) => {
        const percentage = (prob * 100).toFixed(2);
        html += `
            <div class="probability-item">
                <span class="me-2"><strong>Class ${index}:</strong></span>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${percentage}%"></div>
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
