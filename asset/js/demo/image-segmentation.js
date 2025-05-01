const API_URL = 'http://127.0.0.1:5000/segment';
const HEALTH_URL = 'http://127.0.0.1:5000/health';
const EXPECTED_TOKEN = 'fe9def684914b305540615b3bba461fbf8ea58f460c72b1fe128d3cef93fe4f8';
let isServerReady = false;
const overlay = document.getElementById('runtime-overlay');
const statusText = document.getElementById('status-text');
const refreshBtn = document.getElementById('refresh-btn');

// Initial check
checkServerStatus();

const bgToggle = document.getElementById('bgToggle');
const bgColorContainer = document.getElementById('bgColorContainer');

bgToggle.addEventListener('change', () => {
    bgColorContainer.style.display = bgToggle.checked ?
        'inline-flex' :
        'none';
});


document.getElementById('imageUpload').addEventListener('change', function(e) {
    const container = document.getElementById('previewContainer');
    container.innerHTML = ''; // Clear previous previews
    // Create array of promises that resolve in order
    const files = Array.from(e.target.files);
    const readers = files.map(file => {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve({
                content: reader.result,
                index: files.indexOf(file)
            });
            reader.readAsDataURL(file);
        });
    });
    // Create placeholder elements first
    const placeholders = files.map((_, index) => {
        const div = document.createElement('div');
        div.className = 'preview-item';
        div.innerHTML = `
                                                <div class="loading-spinner"></div>`;
        container.appendChild(div);
        return {
            div,
            index
        };
    });
    // Process in parallel but maintain order
    Promise.all(readers).then(results => {
        results.forEach(({
            content,
            index
        }) => {
            const img = document.createElement('img');
            img.src = content;
            placeholders[index].div.innerHTML = '';
            placeholders[index].div.appendChild(img);
        });
    });
});
// Server status check
async function checkServerStatus() {
    try {
        statusText.textContent = "Checking server status...";
        refreshBtn.style.display = 'none';
        const response = await fetch(HEALTH_URL);
        if (!response.ok) throw new Error('Server not responding');
        const data = await response.json();
        if (data.token !== EXPECTED_TOKEN) throw new Error('Security verification failed');
        // Success
        isServerReady = true;
        overlay.style.display = 'none';
        document.querySelector('.main-content').style.display = 'block';
    } catch (error) {
        statusText.textContent = `Error: ${error.message}`;
        refreshBtn.style.display = 'block';
        isServerReady = false;
    }
}
// Image analysis
async function uploadImage() {
    if (!isServerReady) {
        alert('Service connection not established');
        return;
    }

    const files = document.getElementById('imageUpload').files;
    if (!files.length) {
        alert("Please select images first!");
        return;
    }

    try {
        const maskresultContainer = document.getElementById('masksSection');
        const modifiedresultContainer = document.getElementById('modifiedSection');
        const spinner = document.getElementById('result-spinner');
        const saveNote = document.getElementById('saveNote');
        maskresultContainer.innerHTML = ``;
        modifiedresultContainer.innerHTML = ``;
        saveNote.style.display = 'none';
        spinner.style.display = 'block';

        const formData = new FormData();
        for (const file of files) {
            formData.append("images", file);
        }
        if (bgToggle.checked) {
            const hex = colorPicker.value.replace('#', '');
            const r = parseInt(hex.substr(0, 2), 16);
            const g = parseInt(hex.substr(2, 2), 16);
            const b = parseInt(hex.substr(4, 2), 16);
            formData.append('red', r);
            formData.append('green', g);
            formData.append('blue', b);
        }

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error('Analysis failed');
        const data = await response.json();
        maskresultContainer.innerHTML = '';

        // Display segmentation masks
        if (data.masks?.length) {
            const maskLabel = document.createElement('p');
            maskLabel.textContent = 'Segmentation Masks:';
            maskresultContainer.appendChild(maskLabel);

            const maskWrapper = document.createElement('div');
            maskWrapper.className = 'preview-container';
            data.masks.forEach(image => {
                const img = document.createElement('img');
                img.src = `data:image/png;base64,${image.base64}`;
                img.alt = image.filename;
                const wrapper = document.createElement('div');
                wrapper.className = 'preview-item';
                wrapper.appendChild(img);
                maskWrapper.appendChild(wrapper);
            });
            maskresultContainer.appendChild(maskWrapper);
        }

        // Display background-modified images (if provided)
        if (bgToggle.checked && data.bg_modified?.length) {
            modifiedresultContainer.innerHTML = '';
            const bgLabel = document.createElement('p');
            bgLabel.textContent = 'Background-Modified Images:';
            modifiedresultContainer.appendChild(bgLabel);

            const bgWrapper = document.createElement('div');
            bgWrapper.className = 'preview-container';
            data.bg_modified.forEach(image => {
                const img = document.createElement('img');
                img.src = `data:image/png;base64,${image.base64}`;
                img.alt = image.filename;
                const wrapper = document.createElement('div');
                wrapper.className = 'preview-item';
                wrapper.appendChild(img);
                bgWrapper.appendChild(wrapper);
            });
            modifiedresultContainer.appendChild(bgWrapper);
        }

        saveNote.style.display = 'block';
        spinner.style.display = 'none';
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('result').textContent = '❌ Error processing images';
    }
}



// Event listeners
refreshBtn.addEventListener('click', checkServerStatus);