const $ = (id) => document.getElementById(id);
let files = [];
let previewURLs = [];
let ready = false;
let busy = false;
let calibrations = [];

function showError(message = '') {
  $('error').textContent = message;
  $('error').hidden = !message;
}

function updateButton() {
  $('measure').disabled = !ready || busy || files.length === 0;
  $('clear').hidden = files.length === 0;
  $('clear').disabled = busy;
  $('images').disabled = busy;
}

async function checkRuntime() {
  $('refresh').disabled = true;
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error('The local server is unavailable.');
    const status = await response.json();
    ready = status.ready;
    const dot = document.createElement('span');
    dot.className = `dot ${ready ? 'ready' : 'pending'}`;
    $('runtime-status').replaceChildren(dot, document.createTextNode(ready ? 'Ready to measure' : 'Setup needed'));
    $('runtime-detail').textContent = `${status.device.toUpperCase()} · ${status.busy ? 'Measuring' : 'Connected on this machine'}`;
    $('setup-panel').hidden = ready;
    const missing = [];
    if (status.missing_packages.length) missing.push(`Missing packages: ${status.missing_packages.join(', ')}.`);
    if (status.missing_models.length) missing.push(`Missing checkpoints: ${status.missing_models.join(', ')}.`);
    $('setup-reason').textContent = missing.join(' ');
  } catch (error) {
    ready = false;
    $('runtime-status').textContent = 'Runtime unavailable';
    $('runtime-detail').textContent = error.message;
  } finally {
    $('refresh').disabled = false;
    updateButton();
  }
}

function selectFiles(selected) {
  if (busy) return;
  showError();
  const incoming = Array.from(selected);
  if (incoming.length > 12) return showError('Choose up to 12 images per measurement.');
  if (incoming.reduce((size, file) => size + file.size, 0) > 31 * 1024 * 1024) {
    return showError('Choose fewer or smaller images; the upload limit is 32 MB including form data.');
  }
  if (incoming.some((file) => file.type && !file.type.startsWith('image/'))) {
    return showError('Please choose image files.');
  }
  files = incoming;
  calibrations = files.map(() => ({ width_cm: '', height_cm: '' }));
  previewURLs.forEach(URL.revokeObjectURL);
  previewURLs = [];
  $('previews').replaceChildren();
  for (const [index, file] of files.entries()) {
    const figure = document.createElement('figure');
    figure.className = 'preview';
    const image = document.createElement('img');
    image.src = URL.createObjectURL(file);
    previewURLs.push(image.src);
    image.alt = file.name;
    const caption = document.createElement('figcaption');
    caption.textContent = file.name;
    figure.append(image, caption);
    const fields = document.createElement('div');
    fields.className = 'calibration-fields';
    for (const [key, text] of [['width_cm', 'Frame width (cm)'], ['height_cm', 'Frame height (cm)']]) {
      const label = document.createElement('label');
      label.textContent = text;
      const input = document.createElement('input');
      input.type = 'number'; input.min = '0.000001'; input.step = 'any'; input.placeholder = 'Optional';
      input.setAttribute('aria-label', `${file.name}: ${text}`);
      input.addEventListener('input', () => { calibrations[index][key] = input.value; });
      label.append(input); fields.append(label);
    }
    figure.append(fields);
    $('previews').append(figure);
  }
  $('calibration-note').hidden = files.length === 0;
  $('run-status').textContent = files.length ? `${files.length} image${files.length === 1 ? '' : 's'} selected.` : 'Add images to get started.';
  updateButton();
}

function renderResults(data) {
  $('results').replaceChildren();
  $('empty-results').hidden = true;
  $('download-csv').href = data.measurements_csv || data.metadata;
  $('download-csv').hidden = false;
  for (const record of data.images) {
    const card = document.createElement('article');
    card.className = 'result-card';
    // Only constant markup is inserted as HTML; filenames and results use textContent.
    card.innerHTML = '<div class="result-header"><h3></h3><span class="class-badge"></span></div><div class="result-body"><div class="image-view"><select aria-label="Choose output image"></select><div class="interactive-image"></div><img hidden><div class="landmark-info" role="status">Hover or focus a point to see its name, coordinates and measurements. Click to pin.</div></div><div class="measurements"><p class="eyebrow"></p><table><tbody></tbody></table></div></div><div class="result-links"></div>';
    card.querySelector('h3').textContent = record.filename;
    card.querySelector('.class-badge').textContent = record.class;
    card.querySelector('.measurements .eyebrow').textContent = record.calibration ? 'MEASUREMENTS / CM + PIXELS' : 'MEASUREMENTS / PIXELS';
    const image = card.querySelector('.image-view img');
    image.alt = `Landmarks on ${record.filename}`;
    const select = card.querySelector('select');
    select.add(new Option('Interactive landmarks & measurements', 'interactive'));
    const views = [
      ['measurement_image', 'Annotated export'], ['original', 'Original image'],
      ['mask_image', 'Segmentation mask'], ['bg_modified_image', 'Replaced background'],
    ];
    for (const [key, label] of views) {
      if (record.artifacts[key]) select.add(new Option(label, record.artifacts[key]));
    }
    select.addEventListener('change', () => {
      const interactive = select.value === 'interactive';
      image.hidden = interactive;
      card.querySelector('.interactive-image').hidden = !interactive;
      card.querySelector('.landmark-info').hidden = !interactive;
      if (!interactive) image.src = select.value;
    });
    for (const [name, measurement] of Object.entries(record.measurements)) {
      const row = document.createElement('tr');
      row.dataset.measurement = name;
      const label = document.createElement('td');
      label.textContent = name;
      const value = document.createElement('td');
      value.textContent = record.calibration ? `${Number(measurement.distance_cm).toFixed(2)} cm` : `${Number(measurement.distance).toFixed(1)} px`;
      if (record.calibration) {
        const pixels = document.createElement('small');
        pixels.textContent = `${Number(measurement.distance).toFixed(1)} px`;
        value.append(document.createElement('br'), pixels);
      }
      row.append(label, value);
      card.querySelector('tbody').append(row);
    }
    if (!Object.keys(record.measurements).length) {
      const note = document.createElement('p');
      note.className = 'small muted';
      note.textContent = 'Enable derivation for this garment’s custom measurements.';
      card.querySelector('.measurements').append(note);
    }
    window.GarmentViewer.mount(card, record);
    for (const [key, label] of [['measurement_json', 'Download JSON ↗'], ['measurement_image', 'Download annotated image ↗'], ['instruction', 'Instruction used ↗']]) {
      if (!record.artifacts[key]) continue;
      const link = document.createElement('a');
      link.href = record.artifacts[key];
      link.download = '';
      link.textContent = label;
      card.querySelector('.result-links').append(link);
    }
    $('results').append(card);
  }
}

$('refresh').addEventListener('click', checkRuntime);
$('images').addEventListener('change', (event) => selectFiles(event.target.files));
$('clear').addEventListener('click', () => { selectFiles([]); $('images').value = ''; });
$('replace-background').addEventListener('change', (event) => { $('colour-row').hidden = !event.target.checked; });
for (const name of ['dragenter', 'dragover']) {
  $('dropzone').addEventListener(name, (event) => { event.preventDefault(); $('dropzone').classList.add('dragging'); });
}
for (const name of ['dragleave', 'drop']) {
  $('dropzone').addEventListener(name, (event) => { event.preventDefault(); $('dropzone').classList.remove('dragging'); });
}
$('dropzone').addEventListener('drop', (event) => selectFiles(event.dataTransfer.files));
$('measurement-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!ready || busy || !files.length) return;
  showError();
  let calibrationValues;
  try {
    calibrationValues = calibrations.map((entry) => {
      if (!entry.width_cm && !entry.height_cm) return null;
      const width = Number(entry.width_cm), height = Number(entry.height_cm);
      if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        throw new Error('Enter both positive frame width and height in cm for each calibrated image, or leave both blank.');
      }
      return { width_cm: width, height_cm: height };
    });
  } catch (error) { showError(error.message); return; }
  busy = true;
  document.dispatchEvent(new CustomEvent('pipeline-busy', { detail: true }));
  $('previews').querySelectorAll('input').forEach((input) => { input.disabled = true; });
  updateButton();
  $('results-section').setAttribute('aria-busy', 'true');
  $('measurement-form').querySelectorAll('input').forEach((input) => { input.disabled = true; });
  const started = Date.now();
  const updateTime = () => {
    $('run-status').textContent = `Measuring… ${Math.floor((Date.now() - started) / 1000)}s. The first run also loads the models.`;
  };
  updateTime();
  const timer = setInterval(updateTime, 1000);
  try {
    const form = new FormData();
    for (const file of files) form.append('images', file);
    form.append('derive', $('derive').checked);
    form.append('refine', $('refine').checked);
    form.append('calibrations', JSON.stringify(calibrationValues));
    if ($('replace-background').checked) form.append('background', $('background').value);
    const response = await fetch('/api/measure', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Measurement failed.');
    renderResults(data);
    $('run-status').textContent = `Finished ${data.images.length} image${data.images.length === 1 ? '' : 's'} in ${Math.round((Date.now() - started) / 1000)}s.`;
    $('results-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    showError(error.message);
    $('run-status').textContent = 'Measurement stopped. You can adjust the settings and retry.';
  } finally {
    clearInterval(timer);
    busy = false;
    document.dispatchEvent(new CustomEvent('pipeline-busy', { detail: false }));
    $('previews').querySelectorAll('input').forEach((input) => { input.disabled = false; });
    $('results-section').removeAttribute('aria-busy');
    $('measurement-form').querySelectorAll('input').forEach((input) => { input.disabled = false; });
    await checkRuntime();
  }
});
checkRuntime();
