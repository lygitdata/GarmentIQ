/* Responsive SVG overlays retain original image coordinates and support keyboard focus. */
window.GarmentViewer = (() => {
  const ns = 'http://www.w3.org/2000/svg';
  function element(tag, attributes = {}, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text != null) node.textContent = text;
    return node;
  }
  function mount(card, record) {
    const { width, height } = record.image_size;
    const svg = element('svg', { viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': `Interactive landmarks for ${record.filename}` });
    svg.append(element('image', { href: record.artifacts.measurement_base || record.artifacts.original, width, height }));
    const radius = Math.max(width, height) / 150;
    const lines = new Map(), markers = new Map();
    const info = card.querySelector('.landmark-info');
    const rows = [...card.querySelectorAll('tbody tr')];
    let pinned = null;
    const formatLength = (m) => m.distance_cm != null ? `${m.distance_cm.toFixed(2)} cm (${m.distance.toFixed(1)} px)` : `${m.distance.toFixed(1)} px`;
    for (const [name, measurement] of Object.entries(record.measurements)) {
      const a = record.landmarks[measurement.landmarks.start], b = record.landmarks[measurement.landmarks.end];
      const group = element('g', { class: 'measurement-overlay' });
      const line = element('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: 'measurement-line', 'vector-effect': 'non-scaling-stroke' });
      line.append(element('title', {}, `${name}: ${formatLength(measurement)}`));
      group.append(line, element('text', { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - radius, 'font-size': radius * 2, class: 'measurement-label' }, `${name}: ${formatLength(measurement)}`));
      svg.append(group); lines.set(name, group);
    }
    function highlight(names, ids) {
      for (const [name, group] of lines) group.classList.toggle('is-active', names.includes(name));
      for (const [id, marker] of markers) marker.classList.toggle('is-active', ids.includes(id));
      for (const row of rows) row.classList.toggle('is-active', names.includes(row.dataset.measurement));
    }
    function inspect(id) {
      const point = record.landmarks[id];
      const dependent = Object.entries(record.measurements).filter(([, m]) => Object.values(m.landmarks).some((endpoint) => {
        if (endpoint === id) return true;
        const rule = record.landmarks[endpoint]?.derivation;
        return rule && [1, 2, 3, 4, 5].some((i) => rule[`p${i}_id`] === id);
      }));
      highlight(dependent.map(([name]) => name), [id]);
      const heading = document.createElement('strong');
      heading.textContent = `${id} · ${point.description || 'Landmark'} · ${point.predefined ? 'Detected' : 'Derived'}`;
      const coords = document.createElement('p');
      coords.textContent = `x ${point.x.toFixed(2)}, y ${point.y.toFixed(2)} px${point.x_cm != null ? ` · x ${point.x_cm.toFixed(2)}, y ${point.y_cm.toFixed(2)} cm` : ''}${point.conf != null ? ` · confidence ${point.conf.toFixed(3)}` : ''}`;
      const measurements = document.createElement('p');
      measurements.textContent = dependent.length ? dependent.map(([name, m]) => `${name}: ${formatLength(m)}${Object.values(m.landmarks).includes(id) ? '' : ' (via derived point)'}`).join(' · ') : 'No measurements depend on this point.';
      info.replaceChildren(heading, coords, measurements);
      if (point.derivation) {
        const rule = point.derivation;
        const detail = document.createElement('p');
        detail.textContent = `${rule.intersection_mode === 'exact' ? 'Exact intersection' : 'Nearest edge'}: through ${rule.p1_id}, ${rule.direction} to ${rule.p2_id}–${rule.p3_id}; intersect with line ${rule.p4_id}–${rule.p5_id}${rule.intersection_mode === 'exact' ? '.' : ', then locate the nearest contour crossing.'}`;
        info.append(detail);
      }
    }
    function reset() {
      if (pinned) { inspect(pinned); return; }
      highlight([], []);
      info.textContent = 'Hover or focus a point to see its name, coordinates and measurements. Click to pin.';
    }
    for (const [id, point] of Object.entries(record.landmarks)) {
      const marker = element('g', { class: `landmark-marker ${point.predefined ? 'detected' : 'derived'}`, tabindex: '0', role: 'button', 'aria-label': `${id}: ${point.description}, x ${point.x.toFixed(1)}, y ${point.y.toFixed(1)} pixels` });
      marker.append(element('title', {}, `${id}: ${point.description}\nx ${point.x.toFixed(2)}, y ${point.y.toFixed(2)} px`),
        element('circle', { cx: point.x, cy: point.y, r: radius * 2.7, fill: 'transparent', class: 'point-hit-area' }),
        element('circle', { cx: point.x, cy: point.y, r: radius, class: 'point-circle', 'vector-effect': 'non-scaling-stroke' }),
        element('text', { x: point.x + radius * 1.5, y: point.y - radius * 1.3, 'font-size': radius * 2.2, class: 'point-id' }, id));
      marker.addEventListener('pointerenter', () => inspect(id));
      marker.addEventListener('pointerleave', reset);
      marker.addEventListener('focus', () => inspect(id));
      marker.addEventListener('blur', reset);
      const toggle = () => { pinned = pinned === id ? null : id; reset(); };
      marker.addEventListener('click', (event) => { event.stopPropagation(); toggle(); });
      marker.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
        if (event.key === 'Escape') { pinned = null; reset(); }
      });
      markers.set(id, marker); svg.append(marker);
    }
    svg.addEventListener('click', () => { pinned = null; reset(); });
    for (const row of rows) {
      const name = row.dataset.measurement, ids = Object.values(record.measurements[name].landmarks);
      row.tabIndex = 0;
      for (const event of ['pointerenter', 'focus']) row.addEventListener(event, () => highlight([name], ids));
      for (const event of ['pointerleave', 'blur']) row.addEventListener(event, reset);
    }
    card.querySelector('.interactive-image').append(svg);
  }
  return { mount };
})();
