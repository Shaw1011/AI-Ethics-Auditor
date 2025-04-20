// API Endpoints
const API_URL = 'http://localhost:5000/api';
const MODELS_ENDPOINT = `${API_URL}/models`;
const AUDITS_ENDPOINT = `${API_URL}/audits`;
const USERS_ENDPOINT = `${API_URL}/users`;

// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
    // Initialize page based on URL
    const page = window.location.pathname.split('/').pop() || 'index.html';
    
    switch(page) {
        case 'index.html':
            initDashboard();
            break;
        case 'models.html':
            initModelsPage();
            break;
        case 'audits.html':
            initAuditsPage();
            break;
        case 'audit-details.html':
            const auditId = new URLSearchParams(window.location.search).get('id');
            if (auditId) {
                loadAuditDetails(auditId);
            } else {
                showErrorMessage('No audit ID provided');
            }
            break;
        case 'create-audit.html':
            initCreateAuditPage();
            break;
        case 'model-details.html':
            const modelId = new URLSearchParams(window.location.search).get('id');
            if (modelId) {
                loadModelDetails(modelId);
            } else {
                showErrorMessage('No model ID provided');
            }
            break;
    }
});

// Dashboard initialization
function initDashboard() {
    // Fetch recent audits
    fetch(AUDITS_ENDPOINT)
        .then(response => response.json())
        .then(audits => {
            const recentAudits = audits.slice(0, 5);
            const recentAuditsContainer = document.getElementById('recent-audits');
            
            if (recentAuditsContainer) {
                if (recentAudits.length === 0) {
                    recentAuditsContainer.innerHTML = '<p>No audits found. <a href="create-audit.html">Create one</a>.</p>';
                } else {
                    const auditRows = recentAudits.map(audit => `
                        <tr>
                            <td>${audit.id}</td>
                            <td>${audit.name}</td>
                            <td>${audit.status}</td>
                            <td>${new Date(audit.created_at).toLocaleString()}</td>
                            <td>
                                <a href="audit-details.html?id=${audit.id}" class="btn btn-sm">View</a>
                            </td>
                        </tr>
                    `).join('');
                    
                    recentAuditsContainer.innerHTML = `
                        <table>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Name</th>
                                    <th>Status</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${auditRows}
                            </tbody>
                        </table>
                    `;
                }
            }
        })
        .catch(error => {
            console.error('Error fetching audits:', error);
            showErrorMessage('Failed to load recent audits');
        });
    
    // Fetch models count
    fetch(MODELS_ENDPOINT)
        .then(response => response.json())
        .then(models => {
            const modelsCountElement = document.getElementById('models-count');
            if (modelsCountElement) {
                modelsCountElement.textContent = models.length;
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
        });
    
    // Fetch audits count
    fetch(AUDITS_ENDPOINT)
        .then(response => response.json())
        .then(audits => {
            const auditsCountElement = document.getElementById('audits-count');
            if (auditsCountElement) {
                auditsCountElement.textContent = audits.length;
            }
            
            // Count completed audits
            const completedAudits = audits.filter(audit => audit.status === 'completed');
            const completedCountElement = document.getElementById('completed-audits-count');
            if (completedCountElement) {
                completedCountElement.textContent = completedAudits.length;
            }
        })
        .catch(error => {
            console.error('Error fetching audits count:', error);
        });
}

// Models page initialization
function initModelsPage() {
    fetch(MODELS_ENDPOINT)
        .then(response => response.json())
        .then(models => {
            const modelsContainer = document.getElementById('models-list');
            
            if (modelsContainer) {
                if (models.length === 0) {
                    modelsContainer.innerHTML = '<p>No models found. Upload a model to get started.</p>';
                } else {
                    const modelRows = models.map(model => `
                        <tr>
                            <td>${model.id}</td>
                            <td>${model.name}</td>
                            <td>${model.model_type}</td>
                            <td>${model.version || 'N/A'}</td>
                            <td>
                                <a href="model-details.html?id=${model.id}" class="btn btn-sm">View</a>
                                <a href="create-audit.html?model_id=${model.id}" class="btn btn-sm btn-success">Audit</a>
                            </td>
                        </tr>
                    `).join('');
                    
                    modelsContainer.innerHTML = `
                        <table>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Name</th>
                                    <th>Type</th>
                                    <th>Version</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${modelRows}
                            </tbody>
                        </table>
                    `;
                }
            }
            
            // Setup model upload form
            const modelForm = document.getElementById('model-upload-form');
            if (modelForm) {
                modelForm.addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    const formData = {
                        name: document.getElementById('model-name').value,
                        description: document.getElementById('model-description').value,
                        model_type: document.getElementById('model-type').value,
                        version: document.getElementById('model-version').value,
                        metadata: {}
                    };
                    
                    fetch(MODELS_ENDPOINT, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(formData)
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Failed to create model');
                        }
                        return response.json();
                    })
                    .then(data => {
                        showSuccessMessage('Model created successfully!');
                        setTimeout(() => {
                            window.location.reload();
                        }, 1500);
                    })
                    .catch(error => {
                        console.error('Error creating model:', error);
                        showErrorMessage('Failed to create model');
                    });
                });
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            showErrorMessage('Failed to load models');
        });
}

// Audits page initialization
function initAuditsPage() {
    fetch(AUDITS_ENDPOINT)
        .then(response => response.json())
        .then(audits => {
            const auditsContainer = document.getElementById('audits-list');
            
            if (auditsContainer) {
                if (audits.length === 0) {
                    auditsContainer.innerHTML = '<p>No audits found. <a href="create-audit.html">Create one</a>.</p>';
                } else {
                    const auditRows = audits.map(audit => `
                        <tr>
                            <td>${audit.id}</td>
                            <td>${audit.name}</td>
                            <td>${audit.model_id}</td>
                            <td>${audit.status}</td>
                            <td>${new Date(audit.created_at).toLocaleString()}</td>
                            <td>
                                <a href="audit-details.html?id=${audit.id}" class="btn btn-sm">View</a>
                            </td>
                        </tr>
                    `).join('');
                    
                    auditsContainer.innerHTML = `
                        <table>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Name</th>
                                    <th>Model ID</th>
                                    <th>Status</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${auditRows}
                            </tbody>
                        </table>
                    `;
                }
            }
        })
        .catch(error => {
            console.error('Error fetching audits:', error);
            showErrorMessage('Failed to load audits');
        });
}

// Create Audit page initialization
function initCreateAuditPage() {
    // Get available models for dropdown
    fetch(MODELS_ENDPOINT)
        .then(response => response.json())
        .then(models => {
            const modelSelect = document.getElementById('audit-model');
            
            if (modelSelect) {
                if (models.length === 0) {
                    const formContainer = document.getElementById('create-audit-form-container');
                    formContainer.innerHTML = '<div class="alert alert-danger">No models available. Please <a href="models.html">add a model</a> first.</div>';
                } else {
                    // Check if model_id was passed in URL
                    const urlParams = new URLSearchParams(window.location.search);
                    const preselectedModelId = urlParams.get('model_id');
                    
                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = `${model.name} (${model.model_type})`;
                        
                        if (preselectedModelId && preselectedModelId == model.id) {
                            option.selected = true;
                        }
                        
                        modelSelect.appendChild(option);
                    });
                }
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            showErrorMessage('Failed to load models');
        });
    
    // Setup audit form submission
    const auditForm = document.getElementById('create-audit-form');
    if (auditForm) {
        auditForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = {
                name: document.getElementById('audit-name').value,
                description: document.getElementById('audit-description').value,
                model_id: parseInt(document.getElementById('audit-model').value)
            };
            
            fetch(AUDITS_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to create audit');
                }
                return response.json();
            })
            .then(data => {
                showSuccessMessage('Audit created successfully! Running audit in background...');
                setTimeout(() => {
                    window.location.href = 'audits.html';
                }, 2000);
            })
            .catch(error => {
                console.error('Error creating audit:', error);
                showErrorMessage('Failed to create audit');
            });
        });
    }
}

// Load Audit Details
function loadAuditDetails(auditId) {
    fetch(`${AUDITS_ENDPOINT}/${auditId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Audit not found');
            }
            return response.json();
        })
        .then(audit => {
            const auditDetailsContainer = document.getElementById('audit-details');
            const auditMetricsContainer = document.getElementById('audit-metrics');
            
            if (auditDetailsContainer) {
                // Display audit details
                auditDetailsContainer.innerHTML = `
                    <h2>${audit.name}</h2>
                    <p class="audit-description">${audit.description || 'No description provided.'}</p>
                    
                    <div class="audit-meta">
                        <p><strong>Status:</strong> <span class="audit-status status-${audit.status}">${audit.status}</span></p>
                        <p><strong>Model:</strong> ${audit.model_name || `Model #${audit.model_id}`}</p>
                        <p><strong>Created:</strong> ${new Date(audit.created_at).toLocaleString()}</p>
                        <p><strong>Updated:</strong> ${new Date(audit.updated_at).toLocaleString()}</p>
                    </div>
                `;
            }
            
            if (auditMetricsContainer) {
                if (audit.status !== 'completed') {
                    auditMetricsContainer.innerHTML = `
                        <div class="alert alert-warning">
                            <p>This audit is currently in <strong>${audit.status}</strong> status.</p>
                            ${audit.status === 'pending' || audit.status === 'running' ? 
                              '<p>Metrics will be available once the audit is completed.</p>' : 
                              '<p>The audit did not complete successfully. Please try again or check the model configuration.</p>'}
                        </div>
                    `;
                } else if (!audit.metrics || audit.metrics.length === 0) {
                    auditMetricsContainer.innerHTML = '<div class="alert alert-danger">No metrics available for this audit.</div>';
                } else {
                    // Group metrics by category
                    const metricsByCategory = {};
                    audit.metrics.forEach(metric => {
                        if (!metricsByCategory[metric.category]) {
                            metricsByCategory[metric.category] = [];
                        }
                        metricsByCategory[metric.category].push(metric);
                    });
                    
                    // Create metric cards
                    let metricsHtml = '<div class="dashboard-grid">';
                    
                    for (const category in metricsByCategory) {
                        const metrics = metricsByCategory[category];
                        metrics.forEach(metric => {
                            metricsHtml += `
                                <div class="metric-card category-${category.toLowerCase()}">
                                    <div class="metric-title">${metric.name}</div>
                                    <div class="metric-value">${metric.value.toFixed(4)}</div>
                                    <div class="metric-description">
                                        ${metric.details && metric.details.description ? metric.details.description : ''}
                                        ${metric.details && metric.details.interpretation ? `<br><strong>Interpretation:</strong> ${metric.details.interpretation}` : ''}
                                    </div>
                                </div>
                            `;
                        });
                    }
                    
                    metricsHtml += '</div>';
                    auditMetricsContainer.innerHTML = metricsHtml;
                }
            }
        })
        .catch(error => {
            console.error('Error loading audit details:', error);
            showErrorMessage('Failed to load audit details');
        });
}

// Load Model Details
function loadModelDetails(modelId) {
    fetch(`${MODELS_ENDPOINT}/${modelId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Model not found');
            }
            return response.json();
        })
        .then(model => {
            const modelDetailsContainer = document.getElementById('model-details');
            
            if (modelDetailsContainer) {
                // Display model details
                modelDetailsContainer.innerHTML = `
                    <h2>${model.name}</h2>
                    <p class="model-description">${model.description || 'No description provided.'}</p>
                    
                    <div class="model-meta">
                        <p><strong>Type:</strong> ${model.model_type}</p>
                        <p><strong>Version:</strong> ${model.version || 'N/A'}</p>
                        <p><strong>Created:</strong> ${new Date(model.created_at).toLocaleString()}</p>
                        <p><strong>Updated:</strong> ${new Date(model.updated_at).toLocaleString()}</p>
                    </div>
                    
                    <div class="model-actions">
                        <a href="create-audit.html?model_id=${model.id}" class="btn btn-success">Create Audit</a>
                    </div>
                `;
            }
            
            // Fetch audits for this model
            fetch(AUDITS_ENDPOINT)
                .then(response => response.json())
                .then(audits => {
                    const modelAudits = audits.filter(audit => audit.model_id === model.id);
                    const modelAuditsContainer = document.getElementById('model-audits');
                    
                    if (modelAuditsContainer) {
                        if (modelAudits.length === 0) {
                            modelAuditsContainer.innerHTML = '<p>No audits have been performed on this model yet.</p>';
                        } else {
                            const auditRows = modelAudits.map(audit => `
                                <tr>
                                    <td>${audit.id}</td>
                                    <td>${audit.name}</td>
                                    <td>${audit.status}</td>
                                    <td>${new Date(audit.created_at).toLocaleString()}</td>
                                    <td>
                                        <a href="audit-details.html?id=${audit.id}" class="btn btn-sm">View</a>
                                    </td>
                                </tr>
                            `).join('');
                            
                            modelAuditsContainer.innerHTML = `
                                <h3>Audits</h3>
                                <table>
                                    <thead>
                                        <tr>
                                            <th>ID</th>
                                            <th>Name</th>
                                            <th>Status</th>
                                            <th>Created</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${auditRows}
                                    </tbody>
                                </table>
                            `;
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching model audits:', error);
                });
        })
        .catch(error => {
            console.error('Error loading model details:', error);
            showErrorMessage('Failed to load model details');
        });
}

// Utility Functions
function showSuccessMessage(message) {
    const alertContainer = document.createElement('div');
    alertContainer.className = 'alert alert-success';
    alertContainer.textContent = message;
    
    document.body.appendChild(alertContainer);
    
    setTimeout(() => {
        alertContainer.remove();
    }, 3000);
}

function showErrorMessage(message) {
    const alertContainer = document.createElement('div');
    alertContainer.className = 'alert alert-danger';
    alertContainer.textContent = message;
    
    document.body.appendChild(alertContainer);
    
    setTimeout(() => {
        alertContainer.remove();
    }, 3000);
} 