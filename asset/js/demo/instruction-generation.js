const selectedPoints = new Map();

function showGarment() {
  const selected = document.getElementById('garmentSelector').value;

  document.querySelectorAll('svg').forEach(svg => {
    svg.style.display = 'none';
    svg.classList.remove('show');
  });

  // Reset selection UI
  document.querySelectorAll('.point').forEach(c => c.classList.remove('selected'));

  if (selected) {
    const selectedSvg = document.getElementById(selected);
    if (selectedSvg) {
      selectedSvg.style.display = 'block';
      setTimeout(() => selectedSvg.classList.add('show'), 10); // Slight delay to trigger transition
    }

    if (selectedPoints.has(selected)) {
      selectedPoints.get(selected).forEach(id => {
        const circle = document.querySelector(`#${selected} .landmark[data-id="${id}"] circle`);
        if (circle) circle.classList.add('selected');
      });
    }
  }
}

document.querySelectorAll('.landmark').forEach(group => {
  const id = parseInt(group.dataset.id);
  const svgId = group.closest('svg').id;
  const circle = group.querySelector('circle');

  group.addEventListener('click', () => {
    if (!selectedPoints.has(svgId)) selectedPoints.set(svgId, new Set());
    const set = selectedPoints.get(svgId);

    if (set.has(id)) {
      set.delete(id);
      circle.classList.remove('selected');
    } else {
      set.add(id);
      circle.classList.add('selected');
    }
  });
});

function generateJsonBlob() {
  const garment = document.getElementById('garmentSelector').value;
  if (!garment) return;

  const output = {};
  output[garment] = {};

  const svg = document.getElementById(garment);
  svg.querySelectorAll('.landmark').forEach(group => {
    const id = parseInt(group.dataset.id);
    const desc = group.dataset.description;
    if (selectedPoints.has(garment) && selectedPoints.get(garment).has(id)) {
      output[garment][desc] = id;
    }
  });

  // Check if any landmark is selected
  if (Object.keys(output[garment]).length === 0) {
    alert("Please select at least one landmark before exporting.");
    return;
  }

  const blob = new Blob([JSON.stringify(output, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${garment} measurement schema.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function exportToPDF() {
  const garment = document.getElementById('garmentSelector').value;
  if (!garment) return;

  const svg = document.getElementById(garment);
  const clone = svg.cloneNode(true);
  clone.style.display = "block";

  // Serialize SVG to string
  const svgString = new XMLSerializer().serializeToString(clone);
  const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);

  // Check if any landmark is selected
  const selections = selectedPoints.get(garment);
  if (!selections || selections.size === 0) {
    alert("Please select at least one landmark before exporting.");
    return;
  }

  // Create canvas and draw SVG into it
  const canvas = document.createElement("canvas");
  canvas.width = 600;
  canvas.height = 600;
  const ctx = canvas.getContext("2d");

  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const imgData = canvas.toDataURL("image/png");

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF();
    const capitalizedGarment = garment.charAt(0).toUpperCase() + garment.slice(1).replace(/_/g, ' ');

    pdf.setFontSize(16);
    pdf.text(`${capitalizedGarment} Measurement Schema`, 10, 15);
    pdf.addImage(imgData, "PNG", 10, 20, 80, 80);

    // Add selected points list
    let y = 100;
    pdf.setFontSize(10);
    pdf.text("Selected Points to Measure:", 10, y);
    y += 6;

    svg.querySelectorAll('.landmark').forEach(group => {
      const id = parseInt(group.dataset.id);
      const desc = group.dataset.description;
      if (selections && selections.has(id)) {
        pdf.text(`  - ${desc}: ${id}`, 10, y);
        y += 6;
      }
    });

    pdf.save(`${garment} measurement schema.pdf`);
    URL.revokeObjectURL(url);
  };

  img.onerror = () => {
    console.error("Failed to load SVG as image.");
    URL.revokeObjectURL(url);
  };

  img.src = url;
}

function selectAllLandmarks() {
  const garment = document.getElementById('garmentSelector').value;
  if (!garment) return;

  const svg = document.getElementById(garment);
  const landmarks = svg.querySelectorAll('.landmark');
  landmarks.forEach(group => {
    const id = parseInt(group.dataset.id);
    const circle = group.querySelector('circle');
    if (!selectedPoints.has(garment)) selectedPoints.set(garment, new Set());
    const set = selectedPoints.get(garment);
    if (!set.has(id)) {
      set.add(id);
      circle.classList.add('selected');
    }
  });
}

function resetLandmarks() {
  const garment = document.getElementById('garmentSelector').value;
  if (!garment) return;

  const svg = document.getElementById(garment);
  const landmarks = svg.querySelectorAll('.landmark');
  landmarks.forEach(group => {
    const id = parseInt(group.dataset.id);
    const circle = group.querySelector('circle');
    if (selectedPoints.has(garment)) {
      const set = selectedPoints.get(garment);
      set.delete(id);
      circle.classList.remove('selected');
    }
  });
}