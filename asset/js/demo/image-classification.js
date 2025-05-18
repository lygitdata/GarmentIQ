// image-classification.js

document.getElementById('start-loading-model-btn').addEventListener('click', startModelLoading);

let session = null;
let selectedModelUrl = null;
const classes = ['long sleeve dress', 'long sleeve top', 'short sleeve dress',
    'short sleeve top', 'shorts', 'skirt',
    'trousers', 'vest', 'vest dress'
];
const modelChoices = {
    "CNN-3": "https://raw.githubusercontent.com/lygitdata/GarmentIQ/refs/heads/gh-pages/application/demo/image-classification/models/cnn_3.onnx",
    "CNN-4": "https://raw.githubusercontent.com/lygitdata/GarmentIQ/refs/heads/gh-pages/application/demo/image-classification/models/cnn_4.onnx",
    "Tiny ViT": "https://raw.githubusercontent.com/lygitdata/GarmentIQ/refs/heads/gh-pages/application/demo/image-classification/models/tiny_vit.onnx",
    "Tiny ViT (Inditex finetuned)": "https://raw.githubusercontent.com/lygitdata/GarmentIQ/refs/heads/gh-pages/application/demo/image-classification/models/tiny_vit_inditex_finetuned.onnx"
};

const tabsContainer = document.getElementById("model-tabs");
Object.entries(modelChoices).forEach(([modelName, modelUrl]) => {
    const button = document.createElement("button");
    button.className = "btn btn-outline-primary";
    button.innerText = modelName;

    button.onclick = () => {
        // Highlight selected model tab
        document.querySelectorAll("#model-tabs button").forEach(btn => btn.classList.remove("active"));
        button.classList.add("active");

        selectedModelUrl = modelUrl;
        document.getElementById("start-loading-model-btn").disabled = false; // Enable load button
    };

    tabsContainer.appendChild(button);
});

async function startModelLoading() {
    const loadingSection = document.getElementById('model-loading');
    const mainContent = document.getElementById('main-content');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    // Show loading section
    loadingSection.style.display = 'block';
    document.getElementById('start-loading-model-btn').style.display = 'none'; 
    document.getElementById('model-selector').style.display = 'none'; 

    try {
        let modelLoaded = false;
        if (!selectedModelUrl) {
            alert("Please select a model first.");
            return;
        }
        const modelUrl = selectedModelUrl;
        let xhr = new XMLHttpRequest();
        xhr.open('GET', modelUrl, true);
        xhr.responseType = 'arraybuffer';

        xhr.onprogress = (event) => {
            if (event.lengthComputable) {
                let percentLoaded = Math.round((event.loaded / event.total) * 100);
                progressBar.value = percentLoaded;
                progressText.textContent = `${percentLoaded}% loaded`;
            }
        };

        xhr.onload = async function () {
            if (xhr.status === 200) {
                const modelData = xhr.response;
        
                try {
                    // Optional: Set wasm paths if using fallback
                    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
                    // Try WebGL (GPU)
                    session = await ort.InferenceSession.create(modelData, {
                        executionProviders: ['webgl']
                    });
        
                    console.log('✅ ONNX model loaded with WebGL (GPU) backend.');
                } catch (err) {
                    console.warn('⚠️ WebGL backend failed, falling back to CPU (wasm):', err);
        
                    // Fallback to WASM (CPU)
                    session = await ort.InferenceSession.create(modelData, {
                        executionProviders: ['wasm']
                    });
        
                    console.log('✅ ONNX model loaded with WASM (CPU) backend.');
                }
        
                loadingSection.style.display = 'none';
                mainContent.style.display = 'block';
            } else {
                throw new Error('Failed to load model');
            }
        };

        xhr.onerror = function() {
            throw new Error('Failed to fetch model');
        };

        xhr.send();
    } catch (error) {
        console.error('Failed to load ONNX model:', error);
        loadingSection.innerHTML = `<p class="text-danger fw-bold">⚠️ Failed to load model. Please try again later.</p>`;
    }
}



// Set up event listeners for file input and button
const imageInput = document.getElementById('imageUpload');
const previewContainer = document.getElementById('previewContainer');
const resultBox = document.getElementById('result');
const analyzeButton = document.getElementById('analyzeButton');
const spinner = document.getElementById('result-spinner');
let files = []; // Store the files

// Handle image uploads and show previews
imageInput.addEventListener('change', async function(e) {
    const selectedFiles = Array.from(e.target.files);
    
    // Sort files by filename before storing
    selectedFiles.sort((a, b) => a.name.localeCompare(b.name));
    files = selectedFiles; // Store the sorted files

    // Clear preview area
    previewContainer.innerHTML = '';

    // Load and preview images in sorted order
    const imagePromises = selectedFiles.map(file => loadImageFromFile(file));
    const images = await Promise.all(imagePromises);
    
    images.forEach(img => {
        img.classList.add('preview-item');
        previewContainer.appendChild(img);
    });

    // Reset result box text
    resultBox.textContent = 'Ready to analyze images.';
});

// Function to load image from file
async function loadImageFromFile(file) {
    const img = document.createElement('img');
    const reader = new FileReader();
    reader.onload = function(e) {
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
    return new Promise(resolve => {
        img.onload = () => resolve(img);
    });
}

// Analyze images when the button is clicked
analyzeButton.addEventListener('click', async function() {
    if (files.length === 0) {
        resultBox.textContent = 'Please upload images first.';
        return;
    }

    spinner.style.display = 'block';

    resultBox.textContent = 'Analyzing images...';

    const predictions = [];
    for (const file of files) {
        const img = await loadImageFromFile(file);
        const prediction = await runModel(img);
        predictions.push(prediction);
    }

    // Display predictions
    resultBox.textContent = predictions.join(', ');
    spinner.style.display = 'none';
});

// Function to run model on the image
async function runModel(imageElement) {
    if (!session) {
        console.error('ONNX model not loaded yet.');
        return 'Error';
    }

    const inputTensor = preprocessImage(imageElement); // Preprocess the image
    const feeds = {
        input: inputTensor
    }; // Pass the tensor to the model
    const results = await session.run(feeds); // Run the model
    const output = results.output.data; // Get output data
    console.log(output);
    return processOutput(output); // Process and return output
}

// Function to preprocess the image
function preprocessImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 184; // Target image width
    canvas.height = 120; // Target image height
    ctx.drawImage(image, 0, 0, 184, 120);

    const imageData = ctx.getImageData(0, 0, 184, 120);
    const data = imageData.data;
    const floatData = new Float32Array(3 * 120 * 184);

    for (let i = 0; i < 120; i++) {
        for (let j = 0; j < 184; j++) {
            const idx = (i * 184 + j) * 4;
            const r = data[idx] / 255;
            const g = data[idx + 1] / 255;
            const b = data[idx + 2] / 255;

            // Normalize the pixel data
            floatData[i * 184 + j] = (r - 0.8047) / 0.2957; // Normalize Red channel
            floatData[120 * 184 + i * 184 + j] = (g - 0.7808) / 0.3077; // Normalize Green channel
            floatData[2 * 120 * 184 + i * 184 + j] = (b - 0.7769) / 0.3081; // Normalize Blue channel
        }
    }

    return new ort.Tensor('float32', floatData, [1, 3, 120, 184]); // Return tensor for model input
}

// Function to process the output of the model
function processOutput(outputData) {
    const maxIdx = outputData.indexOf(Math.max(...outputData));
    return classes[maxIdx] || `Class ${maxIdx}`; // Return the class with max probability
}
