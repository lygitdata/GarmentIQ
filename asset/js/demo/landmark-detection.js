let baseURL = ''; 
let API_URL = '';
let isServerReady = false;
const EXPECTED_TOKEN = 'a3f7d2c14e65bb2e8f01a9dc4f6c9823d279f1e05b3a6d74c0987b1c2fae3c65';

const overlay = document.getElementById('runtime-overlay');
const statusText = document.getElementById('status-text');
const refreshBtn = document.getElementById('refresh-btn');
const urlInput = document.getElementById('server-url');

refreshBtn.addEventListener('click', async () => {
    baseURL = urlInput.value.trim();
    if (!baseURL) {
        statusText.textContent = 'Please enter a valid server URL.';
        return;
    }

    const healthUrl = `${baseURL}/health`;
    API_URL = `${baseURL}/landmark_detection`
    statusText.textContent = 'Checking server health...';

    try {
        const res = await fetch(healthUrl);
        const data = await res.json();

        if (res.ok && data.token === EXPECTED_TOKEN) {
            isServerReady = true;
            statusText.textContent = 'Connected to server successfully!';
            overlay.style.display = 'none';
            document.querySelector('.main-content').style.display = 'block';
        } else {
            statusText.textContent = 'Server responded, but token is invalid.';
            isServerReady = false;
        }
    } catch (error) {
        console.error('Connection error:', error);
        statusText.textContent = 'Failed to connect to server.';
        isServerReady = false;
    }
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

    const garmentClass = document.getElementById('garmentSelector');
    if (!garmentClass.value) {  // empty string or null check
        alert('Please select a garment type before uploading images.');
        return;  // stop execution here
    }

    try {
        const resultContainer = document.getElementById('landmarkResult');
        const spinner = document.getElementById('result-spinner');
        resultContainer.innerHTML = '';
        spinner.style.display = 'block';

        const formData = new FormData();
        for (const file of files) {
            formData.append("images", file);
        }
        formData.append("garment_class", garmentClass.value);

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error('Analysis failed');
        const data = await response.json();
        resultContainer.innerHTML = '';

        if (data.results?.length) {
            const label = document.createElement('p');
            label.textContent = 'Landmark Detection Results:';
            resultContainer.appendChild(label);

            const wrapper = document.createElement('div');
            wrapper.className = 'preview-container';

            data.results.forEach((base64Img, index) => {
                const img = document.createElement('img');
                img.src = `data:image/png;base64,${base64Img}`;
                img.alt = `Detected landmarks ${index + 1}`;
                const imgWrapper = document.createElement('div');
                imgWrapper.className = 'preview-item-result';
                imgWrapper.appendChild(img);
                wrapper.appendChild(imgWrapper);
            });

            resultContainer.appendChild(wrapper);
        }

        spinner.style.display = 'none';
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('result').textContent = '❌ Error processing images';
    }
}