document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predict-form');
    const resultsSection = document.getElementById('results-section');
    const loadingOverlay = document.getElementById('loading-overlay');
    const predictBtn = document.getElementById('predict-btn');

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading
        loadingOverlay.classList.add('active');
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang xử lý...';
        
        try {
            // Collect form data
            const formData = collectFormData();
            
            // Validate required fields
            if (!validateFormData(formData)) {
                showError('Vui lòng điền đầy đủ các trường bắt buộc (*)');
                return;
            }
            
            // Send prediction request
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                displayResults(result);
            } else {
                showError(result.error || 'Đã xảy ra lỗi khi dự đoán');
            }
            
        } catch (error) {
            console.error('Error:', error);
            showError('Không thể kết nối đến máy chủ. Vui lòng thử lại.');
        } finally {
            loadingOverlay.classList.remove('active');
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<i class="fas fa-stethoscope"></i> Dự đoán nguy cơ';
        }
    });

    // Collect form data
    function collectFormData() {
        const formData = {};
        const inputs = form.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            const name = input.name;
            const value = input.value;
            
            if (name) {
                if (input.type === 'checkbox') {
                    formData[name] = input.checked ? 1 : 0;
                } else if (input.type === 'number' || input.type === 'range') {
                    formData[name] = value !== '' ? parseFloat(value) : null;
                } else {
                    formData[name] = value !== '' ? value : null;
                }
            }
        });
        
        return formData;
    }

    // Validate form data
    function validateFormData(data) {
        const requiredFields = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                               'glucose_fasting', 'glucose_postprandial', 'hba1c'];
        
        for (const field of requiredFields) {
            if (!data[field] || data[field] === '' || data[field] === null) {
                return false;
            }
        }
        
        return true;
    }

    // Display results
    function displayResults(result) {
        const prediction = result.prediction;
        const isDiabetic = prediction.prediction === 1;
        const probability = prediction.probability;
        const confidence = prediction.confidence;
        const message = result.message;
        const advice = result.advice || [];
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Update result badge
        const badge = document.getElementById('result-badge');
        const resultText = document.getElementById('result-text');
        const resultIcon = badge.querySelector('.result-icon i');
        
        if (isDiabetic) {
            badge.className = 'result-badge diabetic';
            resultIcon.className = 'fas fa-exclamation-triangle';
            resultText.textContent = 'Có nguy cơ tiểu đường';
        } else {
            badge.className = 'result-badge non-diabetic';
            resultIcon.className = 'fas fa-check-circle';
            resultText.textContent = 'Không có nguy cơ tiểu đường';
        }
        
        // Update probability bar
        const probabilityFill = document.getElementById('probability-fill');
        const probabilityText = document.getElementById('probability-text');
        const probabilityPercent = Math.round(probability * 100);
        probabilityFill.style.width = probabilityPercent + '%';
        probabilityText.textContent = probabilityPercent + '%';
        
        // Update message
        document.getElementById('result-message').textContent = message;
        
        // Update advice
        const adviceContainer = document.getElementById('advice-container');
        adviceContainer.innerHTML = '';
        
        if (advice && advice.length > 0) {
            advice.forEach((item, index) => {
                const adviceItem = document.createElement('div');
                adviceItem.className = 'advice-item';
                adviceItem.style.animationDelay = (index * 0.1) + 's';
                
                // Determine category class
                const category = item.category || 'Lời khuyên';
                let categoryClass = '';
                if (category.toLowerCase().includes('medical') || category.toLowerCase().includes('chẩn đoán')) {
                    categoryClass = 'medical';
                } else if (category.toLowerCase().includes('prevention') || category.toLowerCase().includes('phòng')) {
                    categoryClass = 'prevention';
                } else if (category.toLowerCase().includes('risk') || category.toLowerCase().includes('nguy cơ')) {
                    categoryClass = 'risk';
                }
                
                adviceItem.className = `advice-item ${categoryClass}`;
                adviceItem.innerHTML = `
                    <div class="advice-category">${category}</div>
                    <div class="advice-text">${item.advice}</div>
                `;
                
                adviceContainer.appendChild(adviceItem);
            });
        } else {
            adviceContainer.innerHTML = `
                <div class="advice-item">
                    <div class="advice-text">Duy trì lối sống lành mạnh, ăn uống cân bằng và tập thể dục thường xuyên.</div>
                </div>
            `;
        }
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Show error
    function showError(message) {
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = `
            background: rgba(231, 76, 60, 0.1);
            color: #e74c3c;
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
            margin-top: 16px;
            font-size: 14px;
        `;
        errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
        
        const formCard = document.querySelector('.form-card');
        formCard.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    // Reset form
    form.querySelector('button[type="reset"]').addEventListener('click', function() {
        resultsSection.style.display = 'none';
        const errorMessage = document.querySelector('.error-message');
        if (errorMessage) {
            errorMessage.remove();
        }
    });

    // Auto-fill example data (for testing)
    const exampleBtn = document.createElement('button');
    exampleBtn.type = 'button';
    exampleBtn.className = 'btn-secondary';
    exampleBtn.style.marginLeft = '12px';
    exampleBtn.innerHTML = '<i class="fas fa-file-medical-alt"></i> Dữ liệu mẫu';
    exampleBtn.onclick = function() {
        const exampleData = {
            age: 45,
            gender: 'Male',
            smoking_status: 'Never',
            alcohol_consumption_per_week: 2,
            physical_activity_minutes_per_week: 120,
            diet_score: 6.5,
            family_history_diabetes: 1,
            hypertension_history: 0,
            cardiovascular_history: 0,
            bmi: 28.5,
            waist_to_hip_ratio: 0.92,
            systolic_bp: 135,
            diastolic_bp: 85,
            heart_rate: 72,
            glucose_fasting: 118,
            glucose_postprandial: 165,
            hba1c: 6.8,
            insulin_level: 10.5,
            cholesterol_total: 190,
            triglycerides: 145,
            ldl_cholesterol: 110,
            hdl_cholesterol: 45
        };
        
        for (const [key, value] of Object.entries(exampleData)) {
            const input = form.querySelector(`[name="${key}"]`);
            if (input) {
                if (input.type === 'checkbox') {
                    input.checked = value === 1;
                } else {
                    input.value = value;
                }
            }
        }
    };
    
    const formActions = document.querySelector('.form-actions');
    formActions.appendChild(exampleBtn);
});