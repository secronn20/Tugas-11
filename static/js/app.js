document.addEventListener('DOMContentLoaded', function() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const selectBtn = document.getElementById('select-btn');
    const uploadPrompt = document.getElementById('upload-prompt');
    const resetBtn = document.getElementById('reset-btn');
    
    const analyzeBtn = document.getElementById('analyze-btn');
    const spinner = document.getElementById('spinner');
    const resultBox = document.getElementById('result-box');
    
    // Result elements
    const resultTitle = document.getElementById('result-title');
    const resultBadge = document.getElementById('result-badge');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const processingTime = document.getElementById('processing-time');
    const predictionTime = document.getElementById('prediction-time');

    let selectedFile = null;

    // Trigger click on input
    selectBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener('click', function() {
        if (!selectedFile) {
            fileInput.click();
        }
    });

    // File input change
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    // Drag and drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
        dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        // Validate type
        if (!file.type.match('image.*')) {
            alert('File harus berupa gambar (JPG, JPEG, PNG)!');
            return;
        }

        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.classList.remove('d-none');
            uploadPrompt.classList.add('d-none');
            analyzeBtn.removeAttribute('disabled');
            resetBtn.classList.remove('d-none');
            
            // Hide previous results
            resultBox.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // Reset button
    resetBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        resetUploader();
    });

    function resetUploader() {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('d-none');
        uploadPrompt.classList.remove('d-none');
        analyzeBtn.setAttribute('disabled', 'true');
        resetBtn.classList.add('d-none');
        resultBox.style.display = 'none';
    }

    // Analyze click (Submit)
    analyzeBtn.addEventListener('click', function() {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        // UI state: loading
        spinner.style.display = 'block';
        resultBox.style.display = 'none';
        analyzeBtn.setAttribute('disabled', 'true');
        resetBtn.classList.add('d-none');

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.message || 'Gagal memproses gambar.') });
            }
            return response.json();
        })
        .then(data => {
            spinner.style.display = 'none';
            analyzeBtn.removeAttribute('disabled');
            resetBtn.classList.remove('d-none');

            if (data.success) {
                // Populate result details
                resultBox.style.display = 'block';
                
                // Set badges and borders
                if (data.result === 'Retak') {
                    resultBox.className = 'glass-card p-4 mt-4 result-box result-glow-positive';
                    resultTitle.innerText = 'TERDETEKSI RETAKAN (RUSAK)';
                    resultBadge.innerText = 'Retak';
                    resultBadge.className = 'status-badge status-positive';
                } else {
                    resultBox.className = 'glass-card p-4 mt-4 result-box result-glow-negative';
                    resultTitle.innerText = 'KONDISI JALAN BAIK (TIDAK RETAK)';
                    resultBadge.innerText = 'Tidak Retak';
                    resultBadge.className = 'status-badge status-negative';
                }

                // Confidence and processing details
                confidenceValue.innerText = data.confidence + '%';
                confidenceBar.style.width = data.confidence + '%';
                processingTime.innerText = data.processing_time;
                predictionTime.innerText = data.timestamp;
                
                // Scroll to result
                resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                alert(data.message || 'Terjadi kesalahan sistem.');
            }
        })
        .catch(error => {
            spinner.style.display = 'none';
            analyzeBtn.removeAttribute('disabled');
            resetBtn.classList.remove('d-none');
            alert(error.message || 'Koneksi gagal atau server bermasalah.');
            console.error(error);
        });
    });
});
