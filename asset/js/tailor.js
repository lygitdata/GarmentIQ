// ------------------------
// Configuration & Globals
// ------------------------
let baseURL = '';
let API_URL = '';
let SETUP_URL = '';
let isServerReady = false;

const EXPECTED_TOKEN = '31f21a4323cf148339669c736f522cf89fa570a2bffc80ba874078477b76f81b';

// ------------------------
// Element References
// ------------------------
const overlay             = document.getElementById('runtime-overlay');
const statusText          = document.getElementById('status-text');
const refreshBtn          = document.getElementById('refresh-btn');
const urlInput            = document.getElementById('server-url');

const doDeriveToggle      = document.getElementById('doDeriveToggle');
const doRefineToggle      = document.getElementById('doRefineToggle');
const bgToggle            = document.getElementById('bgToggle');
const colorPicker         = document.getElementById('colorPicker');
const bgColorContainer    = document.getElementById('bgColorContainer');

const imageUpload         = document.getElementById('imageUpload');

const classificationSection = document.getElementById('classificationSection');
const measureImgSection     = document.getElementById('measureImgSection');
const masksSection          = document.getElementById('masksSection');
const modifiedSection       = document.getElementById('modifiedSection');
const measureJSONSection    = document.getElementById('measureJSONSection');

const spinner  = document.getElementById('result-spinner');
const saveNote = document.getElementById('saveNote');

// ------------------------
// UI Event Listeners
// ------------------------
bgToggle.addEventListener('change', () => {
  bgColorContainer.style.display = bgToggle.checked ? 'inline-flex' : 'none';
});

refreshBtn.addEventListener('click', async () => {
  baseURL   = urlInput.value.trim();
  API_URL   = `${baseURL}/measure`;
  SETUP_URL = `${baseURL}/setup`;

  if (!baseURL) {
    statusText.textContent = 'Please enter a valid server URL.';
    return;
  }

  statusText.textContent = 'Checking server health…';
  try {
    const res  = await fetch(`${baseURL}/health`);
    const data = await res.json();
    if (res.ok && data.token === EXPECTED_TOKEN) {
      isServerReady = true;
      overlay.style.display = 'none';
      document.querySelector('.main-content').style.display = 'block';
      statusText.textContent = 'Connected to server!';
    } else {
      throw new Error('Invalid token');
    }
  } catch (err) {
    console.error(err);
    statusText.textContent = 'Failed to connect to server.';
  }
});

// Preview selected images immediately
imageUpload.addEventListener('change', (e) => {
  const container = document.getElementById('previewContainer');
  container.innerHTML = '';
  Array.from(e.target.files).forEach(file => {
    const reader = new FileReader();
    reader.onload = () => {
      const div = document.createElement('div');
      div.className = 'preview-item';
      div.innerHTML = `<img src="${reader.result}" />`;
      container.appendChild(div);
    };
    reader.readAsDataURL(file);
  });
});

// ------------------------
// Core Functions
// ------------------------

// 1️⃣ Setup Tailor with toggles & optional background color
async function setupTailor() {
  const form = new FormData();
  form.append('do_refine', doRefineToggle.checked ? 'true' : 'false');
  form.append('do_derive', doDeriveToggle.checked ? 'true' : 'false');

  if (bgToggle.checked) {
    // convert hex to RGB
    const hex = colorPicker.value.replace('#','');
    form.append('red',   parseInt(hex.slice(0,2),16));
    form.append('green', parseInt(hex.slice(2,4),16));
    form.append('blue',  parseInt(hex.slice(4,6),16));
  }

  const res = await fetch(SETUP_URL, { method: 'POST', body: form });
  if (!res.ok) throw new Error('Setup failed');
  return res.json();
}

// 2️⃣ Upload images & render results
async function uploadImage() {
  if (!isServerReady) {
    alert('Please connect to the server first.');
    return;
  }
  const files = imageUpload.files;
  if (!files.length) {
    alert('Select at least one image.');
    return;
  }

  try {
    // UI reset
    spinner.style.display = 'block';
    saveNote.style.display = 'none';
    [ classificationSection,
      measureImgSection,
      masksSection,
      modifiedSection,
      measureJSONSection
    ].forEach(sec => sec.innerHTML = '');

    // 1) Call /setup
    await setupTailor();

    // 2) Call /measure
    const form = new FormData();
    for (const f of files) form.append('images', f);
    if (bgToggle.checked) {
      const hex = colorPicker.value.replace('#','');
      form.append('red',   parseInt(hex.slice(0,2),16));
      form.append('green', parseInt(hex.slice(2,4),16));
      form.append('blue',  parseInt(hex.slice(4,6),16));
    }
    const res = await fetch(API_URL, { method: 'POST', body: form });
    if (!res.ok) throw new Error('Measurement failed');
    const results = await res.json();

    // 3️⃣ Render each section

    // --- Classification (always) ---
    const clsTitle = document.createElement('h3');
    clsTitle.textContent = 'Classification Results';
    classificationSection.appendChild(clsTitle);
    results.forEach(r => {
      const p = document.createElement('p');
      p.textContent = `${r["Image name"]}: ${r.Class || '–'}`;
      classificationSection.appendChild(p);
    });

    // --- Measurement Images (always) ---
    const measTitle = document.createElement('h3');
    measTitle.textContent = 'Measurement Images';
    measureImgSection.appendChild(measTitle);
    results.forEach(r => {
      const div = document.createElement('div');
      div.className = 'preview-item';
      div.innerHTML = `<img src="data:image/png;base64,${r["Measurement image"]}" />`;
      measureImgSection.appendChild(div);
    });

    // --- Segmentation Masks (conditional) ---
    const hasMasks = results.some(r => r.Mask);
    if (hasMasks) {
      const maskTitle = document.createElement('h3');
      maskTitle.textContent = 'Segmentation Masks';
      masksSection.appendChild(maskTitle);
      results.forEach(r => {
        if (r.Mask) {
          const div = document.createElement('div');
          div.className = 'preview-item';
          div.innerHTML = `<img src="data:image/png;base64,${r.Mask}" />`;
          masksSection.appendChild(div);
        }
      });
    }

    // --- Background-Modified Images (conditional) ---
    const hasBg = results.some(r => r["Background modified"]);
    if (hasBg) {
      const bgTitle = document.createElement('h3');
      bgTitle.textContent = 'Background-Modified Images';
      modifiedSection.appendChild(bgTitle);
      results.forEach(r => {
        if (r["Background modified"]) {
          const div = document.createElement('div');
          div.className = 'preview-item';
          div.innerHTML = `<img src="data:image/png;base64,${r["Background modified"]}" />`;
          modifiedSection.appendChild(div);
        }
      });
    }

    // --- Measurement JSON Links (always) ---
    const jsonTitle = document.createElement('h3');
    jsonTitle.textContent = 'Measurement JSON';
    measureJSONSection.appendChild(jsonTitle);

    results.forEach(r => {
      const b64 = r["Measurement JSON (base64)"];
      if (b64) {
        // 1) Decode base64 to text
        const jsonText = atob(b64);

        // 2) Create a Blob and object URL
        const blob = new Blob([jsonText], { type: 'application/json' });
        const url  = URL.createObjectURL(blob);

        // 3) Create paragraph + link
        const p = document.createElement('p');
        p.innerHTML = `${r["Image name"]}: `;
        
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.download = `${r["Image name"].split('.').slice(0, -1).join('.')}_measurement.json`;
        a.textContent = 'Open the JSON in a new window';

        p.appendChild(a);
        measureJSONSection.appendChild(p);
      }
    });

    // Done
    spinner.style.display = 'none';
    saveNote.style.display = 'block';

  } catch (err) {
    console.error(err);
    spinner.style.display = 'none';
    alert('Error: ' + err.message);
  }
}

// Hook "Start measurement" button
document.getElementById('start-btn').addEventListener('click', uploadImage);
