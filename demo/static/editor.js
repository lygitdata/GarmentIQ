/* Category-specific authoring, using the website's five-reference derivation schema. */
(() => {
  const get = (id) => document.getElementById(id);
  const ns = 'http://www.w3.org/2000/svg';
  const drafts = new Map();
  let state = null, editingId = null, pointDirty = false, pipelineBusy = false;
  let nextEndpoint = 'start', loadSequence = 0, downloadURL = null;
  function message(text, error = false) {
    get('editor-status').textContent = text;
    get('editor-status').classList.toggle('editor-error', error);
  }
  function guard(action) {
    return (...args) => { try { action(...args); } catch (error) { message(error.message, true); } };
  }
  function changed() { state.dirty = true; message('Unsaved draft. Use “Save & replace category defaults” to apply it to the pipeline.'); }
  function pointLabel(id) { return `${id} · ${state.points[id].description.replaceAll('_', ' ')}${state.points[id].predefined ? '' : ' (derived)'}`; }
  function fillSelect(select, ids, selected) {
    select.replaceChildren(...ids.map((id) => new Option(pointLabel(id), id)));
    if (ids.includes(String(selected))) select.value = selected;
  }
  function svgNode(tag, attributes = {}, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text != null) node.textContent = text;
    return node;
  }
  function lineIntersection(rule) {
    const p = [1, 2, 3, 4, 5].map((i) => state.points[rule[`p${i}_id`]]);
    let dx = p[2].x - p[1].x, dy = p[2].y - p[1].y;
    if (rule.direction === 'perpendicular') [dx, dy] = [-dy, dx];
    const ex = p[4].x - p[3].x, ey = p[4].y - p[3].y;
    const cross = dx * ey - dy * ex;
    if (Math.abs(cross) < 1e-8) return null;
    const t = ((p[3].x - p[0].x) * ey - (p[3].y - p[0].y) * ex) / cross;
    return { x: p[0].x + t * dx, y: p[0].y + t * dy };
  }
  function formRule() {
    const rule = { function: 'derive_keypoint_coord', direction: get('derived-direction').value, intersection_mode: get('derived-mode').value };
    for (let i = 1; i <= 5; i++) rule[`p${i}_id`] = get(`derived-p${i}`).value;
    return rule;
  }
  function drawDiagram() {
    if (!state) return;
    const svg = svgNode('svg', { viewBox: state.diagram.viewBox, role: 'group', 'aria-label': `${state.category} measurement diagram` });
    for (const path of state.diagram.paths) svg.append(svgNode('path', { d: path, class: 'garment-outline' }));
    const start = get('measurement-start').value, end = get('measurement-end').value;
    for (const measurement of Object.values(state.measurements)) {
      const a = state.points[measurement.landmarks.start], b = state.points[measurement.landmarks.end];
      if (a && b) svg.append(svgNode('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: 'editor-measurement-line' }));
    }
    // Reminders are drawn directly on the diagram as well as beside the selects.
    if (editingId && !get('derived-form').hidden) {
      const rule = formRule(), anchor = state.points[rule.p1_id];
      const a = state.points[rule.p2_id], b = state.points[rule.p3_id];
      const c = state.points[rule.p4_id], d = state.points[rule.p5_id];
      if (anchor && a && b && c && d) {
        let dx = b.x-a.x, dy = b.y-a.y;
        if (rule.direction === 'perpendicular') [dx, dy] = [-dy, dx];
        const length = Math.hypot(dx, dy) || 1;
        svg.append(svgNode('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: 'direction-reference' }),
          svgNode('line', { x1: c.x, y1: c.y, x2: d.x, y2: d.y, class: 'intersection-reference' }),
          svgNode('line', { x1: anchor.x-dx/length*300, y1: anchor.y-dy/length*300, x2: anchor.x+dx/length*300, y2: anchor.y+dy/length*300, class: 'derivation-guide' }));
      }
    }
    for (const [id, point] of Object.entries(state.points)) {
      const g = svgNode('g', { tabindex: '0', role: 'button', 'aria-label': pointLabel(id), class: `editor-point ${point.predefined ? 'detected' : 'derived'} ${[start, end].includes(id) ? 'selected' : ''}` });
      g.append(svgNode('title', {}, pointLabel(id)), svgNode('circle', { cx: point.x, cy: point.y, r: 4 }),
        svgNode('text', { x: point.x, y: point.y+1.4 }, id));
      const choose = () => {
        get(`measurement-${nextEndpoint}`).value = id;
        nextEndpoint = nextEndpoint === 'start' ? 'end' : 'start';
        drawDiagram();
      };
      g.addEventListener('click', choose);
      g.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); } });
      svg.append(g);
    }
    get('editor-diagram').replaceChildren(svg);
  }
  function syncMeasurements() {
    if (!state) return;
    const measurements = {};
    for (const row of get('editor-measurements').rows) {
      const name = row.querySelector('[data-field="name"]').value.trim();
      if (!name || measurements[name]) throw new Error('Measurement names must be non-empty and unique.');
      const start = row.querySelector('[data-field="start"]').value, end = row.querySelector('[data-field="end"]').value;
      if (start === end) throw new Error(`Choose different endpoints for “${name}”.`);
      measurements[name] = { landmarks: { start, end }, description: row.querySelector('[data-field="description"]').value };
    }
    state.measurements = measurements;
  }
  function renderMeasurements() {
    const body = get('editor-measurements');
    body.replaceChildren();
    const ids = Object.keys(state.points);
    for (const [name, measurement] of Object.entries(state.measurements)) {
      const row = document.createElement('tr');
      for (const key of ['name', 'start', 'end', 'description']) {
        const cell = document.createElement('td');
        const input = document.createElement(['start', 'end'].includes(key) ? 'select' : 'input');
        input.dataset.field = key;
        input.setAttribute('aria-label', `${name}: ${key}`);
        if (input.tagName === 'SELECT') fillSelect(input, ids, measurement.landmarks[key]);
        else { input.value = key === 'name' ? name : measurement.description || ''; input.maxLength = key === 'name' ? 120 : 500; }
        input.addEventListener('input', changed);
        input.addEventListener('change', guard(() => { syncMeasurements(); drawDiagram(); }));
        cell.append(input); row.append(cell);
      }
      const cell = document.createElement('td'), remove = document.createElement('button');
      remove.type = 'button'; remove.className = 'text-button'; remove.textContent = 'Remove';
      remove.addEventListener('click', guard(() => { row.remove(); syncMeasurements(); changed(); drawDiagram(); }));
      cell.append(remove); row.append(cell); body.append(row);
    }
    fillSelect(get('measurement-start'), ids, get('measurement-start').value);
    fillSelect(get('measurement-end'), ids, get('measurement-end').value || ids[1]);
  }
  function showDerived(id) {
    editingId = id; pointDirty = false;
    get('derived-form').hidden = !id;
    if (!id) { drawDiagram(); return; }
    const point = state.points[id];
    get('derived-name').value = point?.description || '';
    const existing = Object.values(state.points).find((p) => !p.predefined)?.derivation;
    const ids = Object.keys(state.diagram.landmarks);
    const rule = point?.derivation || existing || { p1_id: ids[0], p2_id: ids[1], p3_id: ids[2], p4_id: ids[3], p5_id: ids.at(-1), direction: 'perpendicular' };
    get('derived-direction').value = rule.direction;
    get('derived-mode').value = rule.intersection_mode || 'nearest_edge';
    describeMode();
    for (let i = 1; i <= 5; i++) fillSelect(get(`derived-p${i}`), ids, rule[`p${i}_id`]);
    drawDiagram();
  }
  function describeMode() {
    const exact = get('derived-mode').value === 'exact';
    get('derived-mode-help').textContent = exact
      ? 'Keep the mathematical line–line intersection, even if it is inside or outside the garment.'
      : 'Use the contour crossing nearest the two lines’ intersection.';
    get('rule-explanation').textContent = exact
      ? 'Exact intersection: return the intersection of the line through point 1 (parallel/perpendicular to 2–3) and the line through points 4–5. The mask does not move this point. Reference points must be detected landmarks.'
      : 'Nearest edge: intersect those two lines, then choose the crossing of the line through point 1 and the segmentation contour nearest that intersection. Reference points must be detected landmarks.';
  }
  function renderDerived(selected) {
    const ids = Object.keys(state.points).filter((id) => !state.points[id].predefined);
    const select = get('derived-choice');
    fillSelect(select, ids, selected);
    if (!ids.length) select.add(new Option('No derived landmarks yet', ''));
    showDerived(select.value || null);
  }
  function renderState() {
    nextEndpoint = 'start';
    get('editor-body').hidden = false;
    get('editor-reload').disabled = pipelineBusy;
    renderMeasurements(); renderDerived();
    message(state.dirty ? 'Restored unsaved draft for this category.' : `Loaded ${state.category}. ${Object.keys(state.measurements).length} measurements.`);
    updateDownload();
  }
  function instruction() {
    syncMeasurements();
    // Keep the measurement endpoints, their derived rules, and their detected references.
    const needed = new Set([
      ...Object.values(state.measurements).flatMap((m) => Object.values(m.landmarks)),
      ...Object.keys(state.points).filter((id) => !state.points[id].predefined),
    ]);
    for (const id of [...needed]) {
      const rule = state.points[id]?.derivation;
      if (rule) for (let i = 1; i <= 5; i++) needed.add(rule[`p${i}_id`]);
    }
    const landmarks = Object.fromEntries([...needed].map((id) => [id, state.points[id]]));
    return { [state.category]: { landmarks, measurements: state.measurements } };
  }
  function updateDownload() {
    if (downloadURL) URL.revokeObjectURL(downloadURL);
    downloadURL = URL.createObjectURL(new Blob([JSON.stringify(instruction(), null, 2)], { type: 'application/json' }));
    get('editor-download').href = downloadURL;
    get('editor-download').download = `${state.category}.json`;
  }
  async function loadCategory(category, reload = false) {
    const sequence = ++loadSequence;
    if (!category) { state = null; get('editor-body').hidden = true; get('editor-reload').disabled = true; message('Choose a garment category to begin.'); return; }
    message('Loading category instructions…');
    get('editor-body').hidden = true;
    try {
      if (!reload && drafts.has(category)) { state = drafts.get(category); renderState(); return; }
      const response = await fetch(`/api/instructions/${encodeURIComponent(category)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      if (sequence !== loadSequence) return;
      const saved = data.instruction[category];
      state = { category, revision: data.revision, diagram: data.diagram, dirty: false,
        points: { ...structuredClone(data.diagram.landmarks), ...saved.landmarks }, measurements: saved.measurements };
      drafts.set(category, state); renderState();
    } catch (error) { message(error.message, true); }
  }
  get('editor-category').addEventListener('change', guard(() => {
    if (state) {
      if (pointDirty) { get('editor-category').value = state.category; throw new Error('Apply your derived-point edits before changing category, or reload from disk to discard them.'); }
      try { syncMeasurements(); } catch (error) { get('editor-category').value = state.category; throw error; }
    }
    loadCategory(get('editor-category').value);
  }));
  get('editor-reload').addEventListener('click', () => loadCategory(get('editor-category').value, true));
  get('derived-choice').addEventListener('change', guard(() => {
    if (pointDirty) { get('derived-choice').value = editingId; throw new Error('Apply your point edits before choosing another derived point.'); }
    showDerived(get('derived-choice').value);
  }));
  get('derived-new').addEventListener('click', guard(() => {
    if (pointDirty) throw new Error('Apply your current derived point first.');
    const id = String(Math.max(...Object.keys(state.points).map(Number)) + 1);
    get('derived-choice').add(new Option(`${id} · New derived point`, id));
    get('derived-choice').value = id; showDerived(id); pointDirty = true;
    get('derived-name').focus();
    message('Name the new point and choose its five references, then apply it to the draft.');
  }));
  get('derived-form').addEventListener('input', () => { pointDirty = true; describeMode(); drawDiagram(); });
  get('derived-form').addEventListener('submit', guard((event) => {
    event.preventDefault(); syncMeasurements();
    const rule = formRule();
    const position = lineIntersection(rule);
    if (rule.p2_id === rule.p3_id || rule.p4_id === rule.p5_id || !position) {
      throw new Error('These diagram reference lines are parallel or use the same point twice. Choose different reference pairs.');
    }
    const description = get('derived-name').value.trim();
    if (!description) throw new Error('Give the derived point a name.');
    state.points[editingId] = { predefined: false, description, ...position, derivation: rule };
    pointDirty = false; changed(); renderMeasurements(); renderDerived(editingId); updateDownload();
  }));
  get('derived-remove').addEventListener('click', guard(() => {
    syncMeasurements();
    delete state.points[editingId];
    state.measurements = Object.fromEntries(Object.entries(state.measurements).filter(([, m]) => !Object.values(m.landmarks).includes(editingId)));
    pointDirty = false; changed(); renderMeasurements(); renderDerived();
  }));
  get('new-measurement').addEventListener('submit', guard((event) => {
    event.preventDefault(); syncMeasurements();
    const name = get('measurement-name').value.trim();
    const start = get('measurement-start').value, end = get('measurement-end').value;
    if (!name || state.measurements[name]) throw new Error('Choose a non-empty, unique measurement name.');
    if (start === end) throw new Error('Choose two different measurement endpoints.');
    state.measurements[name] = { landmarks: { start, end }, description: '' };
    get('measurement-name').value = ''; changed(); renderMeasurements(); drawDiagram(); updateDownload();
  }));
  for (const id of ['measurement-start', 'measurement-end']) get(id).addEventListener('change', drawDiagram);
  get('editor-download').addEventListener('click', guard((event) => {
    try {
      if (pointDirty) throw new Error('Apply your derived-point edits before downloading instructions.');
      updateDownload();
    } catch (error) { event.preventDefault(); throw error; }
  }));
  get('editor-save').addEventListener('click', async () => {
    if (!state || pipelineBusy) return;
    get('editor-save').disabled = true;
    try {
      if (state.category !== get('editor-category').value) throw new Error('Wait for the selected category to finish loading.');
      if (pointDirty) throw new Error('Apply the derived-point edits to your draft before saving.');
      const response = await fetch(`/api/instructions/${encodeURIComponent(state.category)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction: instruction(), revision: state.revision }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      state.revision = data.revision; state.dirty = false;
      if (Object.values(data.instruction[state.category].landmarks).some((p) => !p.predefined)) get('derive').checked = true;
      message(`Saved src/garmentiq/instruction/${state.category}.json. The next ${state.category} measurement uses these defaults. Previous instructions backed up locally.`);
      updateDownload();
    } catch (error) { message(error.message, true); }
    finally { get('editor-save').disabled = pipelineBusy; }
  });
  document.addEventListener('pipeline-busy', (event) => {
    pipelineBusy = event.detail;
    get('instruction-editor').querySelectorAll('input, select, button').forEach((control) => { control.disabled = pipelineBusy; });
    get('editor-reload').disabled = pipelineBusy || !state;
  });
  fetch('/api/instructions').then(async (response) => {
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    for (const category of data.categories) get('editor-category').add(new Option(category, category));
  }).catch((error) => message(`Could not load categories: ${error.message}`, true));
})();
