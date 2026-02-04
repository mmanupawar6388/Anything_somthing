// Get DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

// Clear button
clearBtn.addEventListener('click', () => {
    resetUI();
});

function handleFile(file) {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Send to server
    predictImage(file);
}

function predictImage(file) {
    // Hide previous results and errors
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    loading.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            loading.style.display = 'none';

            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'Prediction failed');
            }
        })
        .catch(error => {
            loading.style.display = 'none';
            showError('Network error: ' + error.message);
        });
}

function displayResults(data) {
    // Show top prediction
    document.getElementById('topClass').textContent = data.prediction;
    document.getElementById('topConfidence').textContent =
        (data.confidence * 100).toFixed(1) + '%';

    // Show all predictions
    const predictionBars = document.getElementById('predictionBars');
    predictionBars.innerHTML = '';

    for (const [className, confidence] of Object.entries(data.all_predictions)) {
        const barDiv = document.createElement('div');
        barDiv.className = 'prediction-bar';

        const percentage = (confidence * 100).toFixed(1);

        barDiv.innerHTML = `
            <div class="bar-label">
                <span class="bar-name">${className}</span>
                <span class="bar-value">${percentage}%</span>
            </div>
            <div class="bar-container">
                <div class="bar-fill" style="width: ${percentage}%"></div>
            </div>
        `;

        predictionBars.appendChild(barDiv);
    }

    resultsSection.style.display = 'block';
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'block';
}

function resetUI() {
    fileInput.value = '';
    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    loading.style.display = 'none';
}
