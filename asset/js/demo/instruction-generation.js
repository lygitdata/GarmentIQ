const garmentSelector = document.getElementById('garmentSelector');
const garmentDisplayContainer = document.getElementById('garmentDisplayContainer');
const selectedPoints = new Map();
let savedSelection = {};
const toggleBtn = document.getElementById('toggleCustomLandmarks');
let customLandmarkMode = false;
toggleBtn.addEventListener('click', toggleCustomLandmarksMode);

// Helper functions
function nextButton(currentElement, nextElement) {
	const current = document.getElementById(currentElement);
	const next = document.getElementById(nextElement);

	current.style.display = 'none';
	next.style.display = 'block';
}

function backButton(currentElement, backElement) {
	const current = document.getElementById(currentElement);
	const back = document.getElementById(backElement);

	current.style.display = 'none';
	back.style.display = 'block';
}

function getActiveSvg() {
	return garmentDisplayContainer.querySelector('svg.show');
}

function applyTransform() {
	const activeSvg = getActiveSvg();
	if (activeSvg) {
		activeSvg.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
	}
}

function screenToSvgCoords(screenX, screenY) {
	return {
		x: (screenX - offsetX) / scale,
		y: (screenY - offsetY) / scale
	};
}

// Function(s) for garmentPreviewContainer
function previewGarment() {
	const selectedGarmentId = garmentSelector.value;

	if (selectedGarmentId) {
		const formattedGarmentId = selectedGarmentId.replace(/ /g, "_");
		const garmentPreviewContainer = document.getElementById("garmentPreview");

		garmentPreviewContainer.innerHTML = '';

		const svgImage = document.createElement("img");
		svgImage.src = `https://garmentiq.ly.gd.edu.kg/asset/img/garment_example/${formattedGarmentId}.svg`;
		svgImage.alt = selectedGarmentId;

		svgImage.style.width = "auto";
		svgImage.style.height = "325px";

		garmentPreviewContainer.appendChild(svgImage);
		document.getElementById('garmentPreviewContainerNextButton').style.display = "block";
	} else {
		console.log("No garment selected.")
	}
}

function backToSelectEmpty() {
	selectedPoints.clear();
}

// Function(s) for garmentLandmarkContainer
function showGarmentToSelect() {
	scale = 1;
	offsetX = 0;
	offsetY = 0;

	applyTransform();

	const selectedGarmentId = garmentSelector.value;

	if (selectedGarmentId) {
		const allSvgs = document.querySelectorAll('#garmentDisplayContainer svg');
		allSvgs.forEach(svg => {
			svg.style.display = 'none';
		});

		const selectedSvg = document.getElementById(selectedGarmentId);
		if (selectedSvg) {
			selectedSvg.style.display = 'block';
			selectedSvg.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
			setTimeout(() => {
				selectedSvg.classList.add('show');
			}, 10);

			const currentSelections = selectedPoints.get(selectedGarmentId);
			if (currentSelections) {
				currentSelections.forEach(id => {
					const circle = selectedSvg.querySelector(`.landmark[data-id="${id}"] circle.point`);
					if (circle) {
						circle.classList.add('selected');
					}
				});
			}
		} else {
			console.warn(`SVG element with ID "${selectedGarmentId}" not found.`);
		}
	} else {
		console.log("No garment selected.");
	}
}

function selectAllLandmarks() {
	const garment = garmentSelector.value;
	if (!garment) return;

	const svg = document.getElementById(garment);
	if (!svg) return;

	if (!selectedPoints.has(garment)) {
		selectedPoints.set(garment, new Set());
	}
	const currentSelections = selectedPoints.get(garment);

	svg.querySelectorAll('.landmark').forEach(group => {
		const id = parseInt(group.dataset.id);
		const circle = group.querySelector('circle.point');
		if (!currentSelections.has(id)) {
			currentSelections.add(id);
			if (circle) circle.classList.add('selected');
		}
	});
}

function resetLandmarks() {
	const garment = garmentSelector.value;
	if (!garment) return;

	const svg = document.getElementById(garment);
	if (!svg) return;

	const currentSelections = selectedPoints.get(garment);
	if (currentSelections) {
		const idsToRemove = Array.from(currentSelections);

		idsToRemove.forEach(id => {
			const circle = svg.querySelector(`.landmark[data-id="${id}"] circle.point`);
			if (circle) {
				circle.classList.remove('selected');
			}
			currentSelections.delete(id);
		});
	}
}

garmentDisplayContainer.addEventListener('click', (e) => {
	const landmarkGroup = e.target.closest('.landmark');
	if (!landmarkGroup) return;

	const circle = landmarkGroup.querySelector('circle.point');
	if (!circle || !landmarkGroup.contains(e.target)) return;

	const id = parseInt(landmarkGroup.dataset.id);
	const svg = landmarkGroup.closest('svg');
	if (!svg) return;
	const svgId = svg.id;

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

function toggleCustomLandmarksMode() {
	const svgId = garmentSelector.value;
	const svg = document.getElementById(svgId);
	if (!svg) return console.error(`SVG "#${svgId}" not found`);

	customLandmarkMode = !customLandmarkMode;
	toggleBtn.textContent = `Custom Landmarks Mode (${customLandmarkMode ? 'On' : 'Off'})`;
	svg.style.cursor = customLandmarkMode ? 'crosshair' : 'default';

	if (customLandmarkMode)
		svg.addEventListener('click', handleCustomLandmarkClick);
	else
		svg.removeEventListener('click', handleCustomLandmarkClick);
}

function handleCustomLandmarkClick(event) {
	event.preventDefault();
	const svg = event.currentTarget;

	const pt = svg.createSVGPoint();
	pt.x = event.clientX;
	pt.y = event.clientY;
	const {
		x,
		y
	} = pt.matrixTransform(svg.getScreenCTM().inverse());

	const clickedG = event.target.closest('g.landmark[data-custom="true"]');
	if (clickedG) {
		removeCustom(clickedG, svg);
	} else if (!event.target.closest('g.landmark')) {
		addCustom(x, y, svg);
	}
}

function addCustom(x, y, svg) {
	const predefinedCount = svg.querySelectorAll('g.landmark:not([data-custom])').length;
	const customCount = svg.querySelectorAll('g.landmark[data-custom]').length;
	const newId = predefinedCount + customCount + 1;
	const desc = `custom_landmark`;

	// Create the <g>
	const g = document.createElementNS(svg.namespaceURI, 'g');
	g.setAttribute('class', 'landmark');
	g.setAttribute('data-id', newId);
	g.setAttribute('data-custom', 'true');
	g.setAttribute('data-description', desc);

	// Circle
	const c = document.createElementNS(svg.namespaceURI, 'circle');
	c.setAttribute('class', 'point');
	c.setAttribute('cx', x);
	c.setAttribute('cy', y);
	c.setAttribute('r', '3.5');

	const t = document.createElementNS(svg.namespaceURI, 'text');
	t.setAttribute('class', 'label');
	t.setAttribute('x', x);
	t.setAttribute('y', y);
	t.textContent = newId;

	g.append(c, t);
	svg.appendChild(g);

	addDescriptionListItem(newId, desc);
}

function removeCustom(g, svg) {
	const removedId = Number(g.dataset.id);
	g.remove();
	removeDescriptionListItem(removedId);

	const laterGs = Array.from(svg.querySelectorAll('g.landmark[data-custom]'))
		.filter(el => Number(el.dataset.id) > removedId)
		.sort((a, b) => Number(a.dataset.id) - Number(b.dataset.id));
	laterGs.forEach(el => {
		const oldId = Number(el.dataset.id);
		const newId = oldId - 1;
		el.dataset.id = newId;
		el.querySelector('text.label').textContent = newId;
	});

	renumberDescriptionListItemsAfter(removedId);
}

function removeAllCustom() {
	const svgId = garmentSelector.value;
	const svg = document.getElementById(svgId);
	if (!svg) return console.error(`SVG "#${svgId}" not found`);

	const customGs = svg.querySelectorAll('g.landmark[data-custom="true"]');
	if (customGs.length === 0) return;

	customGs.forEach(g => {
		const id = Number(g.dataset.id);
		g.remove();
		removeDescriptionListItem(id);
	});

	const remainingCustomGs = Array.from(svg.querySelectorAll('g.landmark[data-custom="true"]'))
		.sort((a, b) => Number(a.dataset.id) - Number(b.dataset.id));

	remainingCustomGs.forEach((g, index) => {
		const newId = index + 1;
		g.dataset.id = newId;
		g.querySelector('text.label').textContent = newId;
	});

	resetDescriptionListItems();
}

function addDescriptionListItem(id, description) {
	const ul = document.getElementById('garmentCustomLandmarkNameList');
	const li = document.createElement('li');
	li.dataset.id = id;

	const label = document.createElement('label');
	label.htmlFor = `desc-${id}`;
	label.textContent = `${id}: `;

	const input = document.createElement('input');
	input.type = 'text';
	input.id = `desc-${id}`;
	input.value = description;

	input.addEventListener('input', e => {
		const uid = Number(e.target.closest('li').dataset.id);
		const svg = document.getElementById(garmentSelector.value);
		const g = svg.querySelector(`g.landmark[data-id="${uid}"][data-custom="true"]`);
		if (g) g.dataset.description = e.target.value;
	});

	li.append(label, input);
	ul.appendChild(li);
}

function removeDescriptionListItem(id) {
	const ul = document.getElementById('garmentCustomLandmarkNameList');
	const li = ul.querySelector(`li[data-id="${id}"]`);
	if (li) li.remove();
}

function renumberDescriptionListItemsAfter(removedId) {
	const ul = document.getElementById('garmentCustomLandmarkNameList');
	const lis = Array.from(ul.querySelectorAll('li'))
		.filter(li => Number(li.dataset.id) > removedId)
		.sort((a, b) => Number(a.dataset.id) - Number(b.dataset.id));

	lis.forEach(li => {
		const oldId = Number(li.dataset.id);
		const newId = oldId - 1;
		li.dataset.id = newId;

		const label = li.querySelector('label');
		label.textContent = `${newId}: `;
		label.htmlFor = `desc-${newId}`;
		const input = li.querySelector('input');
		input.id = `desc-${newId}`;
	});
}

function _getCoords(g) {
	const c = g.querySelector('circle.point');
	return [
		parseFloat(c.getAttribute('cx')),
		parseFloat(c.getAttribute('cy'))
	];
}

function _nearestPredefined(svg, customG, N = 2) {
	const [x0, y0] = _getCoords(customG);
	const predefs = Array.from(svg.querySelectorAll('g.landmark:not([data-custom="true"])'));
	const withDist = predefs.map(g => {
		const [x, y] = _getCoords(g);
		const dx = x - x0,
			dy = y - y0;
		return {
			id: g.dataset.id,
			d2: dx * dx + dy * dy
		};
	});
	return withDist
		.sort((a, b) => a.d2 - b.d2)
		.slice(0, N)
		.map(o => o.id);
}

function saveSelection() {
	savedSelection = {};
	console.log(selectedPoints);
	selectedPoints.forEach((idSet, garmentName) => {
		const svg = document.getElementById(garmentName);
		if (!svg) {
			console.warn(`SVG "#${garmentName}" not found — skipping`);
			return;
		}

		savedSelection[garmentName] = {
			landmarks: {},
			measurements: {}
		};

		const sortedIds = Array.from(idSet)
			.map(Number)
			.sort((a, b) => a - b)
			.map(String);

		sortedIds.forEach(id => {
			const g = svg.querySelector(`g.landmark[data-id="${id}"]`);
			if (!g) return;

			const isCustom = g.dataset.custom === 'true';
			const predefined = !isCustom;
			const description = g.dataset.description || '';
			const [x, y] = _getCoords(g);

			const entry = {
				predefined,
				description,
				x,
				y
			};

			if (isCustom) {
				const neighborIds = _nearestPredefined(svg, g, 2);
				entry.neighbors = {};
				neighborIds.forEach(nid => {
					const ng = svg.querySelector(`g.landmark[data-id="${nid}"]`);
					const [nx, ny] = _getCoords(ng);
					entry.neighbors[nid] = {
						predefined: true,
						description: ng.dataset.description || '',
						x: nx,
						y: ny
					};
				});
			}

			savedSelection[garmentName].landmarks[id] = entry;
		});
	});
	console.log(savedSelection);
}

// Function(s) for garmentMeasurementContainer
function showGarmentWithSelection() {
	const selectedGarmentId = garmentSelector.value;
	const container = document.getElementById('garmentMeasurementDisplayContainer');
	if (!container) {
		console.error('#garmentMeasurementDisplayContainer not found');
		return;
	}

	container.querySelectorAll('svg').forEach(svg => {
		svg.style.display = svg.id === selectedGarmentId ? '' : 'none';
	});

	const svg = container.querySelector(`svg[id="${selectedGarmentId}"]`);
	const landmarkData = (savedSelection[selectedGarmentId] || {}).landmarks;
	if (!svg || !landmarkData) return;

	const ns = svg.namespaceURI;

	svg.querySelectorAll('g.landmark').forEach(g => g.remove());

	Object.keys(landmarkData)
		.map(id => Number(id))
		.sort((a, b) => a - b)
		.forEach(idNum => {
			const id = String(idNum);
			const info = landmarkData[id];

			// group
			const g = document.createElementNS(ns, 'g');
			g.setAttribute('class', 'landmark');
			g.dataset.id = id;
			g.dataset.description = info.description;
			if (!info.predefined) g.dataset.custom = 'true';

			// circle
			const c = document.createElementNS(ns, 'circle');
			c.setAttribute('class', 'point');
			c.setAttribute('cx', info.x);
			c.setAttribute('cy', info.y);
			c.setAttribute('r', 3.5);

			// label
			const t = document.createElementNS(ns, 'text');
			t.setAttribute('class', 'label');
			t.setAttribute('x', info.x);
			t.setAttribute('y', info.y);
			t.textContent = id;

			g.append(c, t);
			svg.appendChild(g);
		});
}

function generateMeasurementDetailTable() {
	const garmentName = garmentSelector.value;
	const landmarkData = savedSelection[garmentName]?.landmarks;
	if (!landmarkData) return;

	// get sorted list of IDs
	const ids = Object.keys(landmarkData)
		.map(Number)
		.sort((a, b) => a - b)
		.map(String);

	const table = document.getElementById('garmentMeasurementDetailTable');

	const container = document.getElementById('garmentMeasurementDisplayContainer');
	const currentSvg = container.querySelector(`svg[id="${garmentName}"]`);
	if (currentSvg) {
		currentSvg.querySelectorAll('line[id^="line-"]').forEach(line => line.remove());
	}

	// remove all rows except the header
	while (table.rows.length > 1) {
		table.deleteRow(1);
	}

	// for each pair (i < j) add a row
	for (let i = 0; i < ids.length; i++) {
		for (let j = i + 1; j < ids.length; j++) {
			const start = ids[i],
				end = ids[j];
			const row = table.insertRow(-1);

			// Measure checkbox
			const cellChk = row.insertCell();
			const chk = document.createElement('input');
			chk.type = 'checkbox';
			chk.dataset.start = start;
			chk.dataset.end = end;
			chk.addEventListener('change', onMeasureToggle);
			cellChk.appendChild(chk);

			// Start, End
			row.insertCell().textContent = start;
			row.insertCell().textContent = end;

			// Name input
			const cellName = row.insertCell();
			const nameInput = document.createElement('input');
			nameInput.type = 'text';
			nameInput.placeholder = 'Name';
			cellName.appendChild(nameInput);

			// Description input
			const cellDesc = row.insertCell();
			const descInput = document.createElement('input');
			descInput.type = 'text';
			descInput.placeholder = 'Description';
			cellDesc.appendChild(descInput);
		}
	}
}

function onMeasureToggle(e) {
	const chk = e.target;
	const start = chk.dataset.start;
	const end = chk.dataset.end;
	const garmentName = garmentSelector.value;

	if (chk.checked) {
		addMeasurementLine(garmentName, start, end);
	} else {
		removeMeasurementLine(garmentName, start, end);
	}
}

function addMeasurementLine(garmentName, start, end) {
	const container = document.getElementById('garmentMeasurementDisplayContainer');
	const svg = container.querySelector(`svg[id="${garmentName}"]`);
	if (!svg) return;

	const getCircle = id => svg.querySelector(`g.landmark[data-id="${id}"] circle.point`);
	const c1 = getCircle(start),
		c2 = getCircle(end);
	if (!c1 || !c2) return;

	const x1 = c1.cx.baseVal.value,
		y1 = c1.cy.baseVal.value;
	const x2 = c2.cx.baseVal.value,
		y2 = c2.cy.baseVal.value;

	// ensure <g id="measurementLines">
	let linesG = svg.querySelector('g#measurementLines');
	if (!linesG) {
		linesG = document.createElementNS(svg.namespaceURI, 'g');
		linesG.id = 'measurementLines';
		svg.appendChild(linesG);
	}

	// create the <line>
	const line = document.createElementNS(svg.namespaceURI, 'line');
	line.id = `line-${start}-${end}`;
	line.setAttribute('x1', x1);
	line.setAttribute('y1', y1);
	line.setAttribute('x2', x2);
	line.setAttribute('y2', y2);
	line.setAttribute('stroke', 'blue');
	line.setAttribute('stroke-dasharray', '4');
	linesG.appendChild(line);
}

function removeMeasurementLine(garmentName, start, end) {
	const container = document.getElementById('garmentMeasurementDisplayContainer');
	const svg = container.querySelector(`svg[id="${garmentName}"]`);
	if (!svg) return;
	const linesG = svg.querySelector('g#measurementLines');
	if (!linesG) return;
	const line = linesG.querySelector(`#line-${start}-${end}`);
	if (line) line.remove();
}

function exportMeasurementsAsJSON() {
	const garmentName = garmentSelector.value;
	const garmentData = savedSelection[garmentName];
	if (!garmentData) {
		alert('No garment selected.');
		return;
	}

	const table = document.getElementById('garmentMeasurementDetailTable');
	const measurements = {};

	for (let i = 1; i < table.rows.length; i++) {
		const row = table.rows[i];
		const chk = row.cells[0].querySelector('input[type="checkbox"]');
		if (!chk || !chk.checked) continue; 
		const start = row.cells[1].textContent.trim();
		const end = row.cells[2].textContent.trim();
		const nameInput = row.cells[3].querySelector('input');
		const descInput = row.cells[4].querySelector('input');

		const name = nameInput.value.trim();
		const description = descInput.value.trim();

		if (!name || !description) {
			alert('Please fill out the Name and Description for all checked measurements.');
			return;
		}

		// Add this measurement
		measurements[name] = {
			landmarks: {
				start: start,
				end: end
			},
			description: description
		};
	}

	garmentData.measurements = measurements;

	const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(savedSelection, null, 2));
	const downloadAnchorNode = document.createElement('a');
	downloadAnchorNode.setAttribute("href", dataStr);
	downloadAnchorNode.setAttribute("download", garmentName + ".json"); // filename = garment id
	document.body.appendChild(downloadAnchorNode); // required for Firefox
	downloadAnchorNode.click();
	downloadAnchorNode.remove();
}

async function exportMeasurementsAsPDF() {
	const garment = garmentSelector.value;
	if (!garment) {
		alert("Please choose a garment first.");
		return;
	}
	const container = document.getElementById('garmentMeasurementDisplayContainer');
	const svg = container.querySelector(`svg[id="${garment}"]`);
	if (!svg) {
		console.error(`Cannot find SVG for garment: ${garment}`);
		return;
	}

	const selections = selectedPoints.get(garment);
	if (!selections || selections.size === 0) {
		alert("Please select at least one landmark before exporting PDF.");
		return;
	}

	const clone = svg.cloneNode(true);
	clone.style.transform = 'none';
	selections.forEach(id => {
		const c = clone.querySelector(`.landmark[data-id="${id}"] circle.point`);
		if (c) c.classList.add('selected');
	});
	const svgString = new XMLSerializer().serializeToString(clone);

	const {
		jsPDF
	} = window.jspdf;
	const pdf = new jsPDF({
		unit: 'mm',
		format: 'letter'
	});
	const pageW = pdf.internal.pageSize.getWidth();
	const pageH = pdf.internal.pageSize.getHeight();
	const margin = 15;
	let yPos = margin;

	pdf.setFontSize(16);
	pdf.text(`Measurement instruction: ${garment}`, margin, yPos + 5);
	yPos += 10;
	pdf.setFontSize(10);
	const now = new Date();
	const formattedDate = now.toLocaleString('en-US', {
	  year: 'numeric',
	  month: '2-digit',
	  day: '2-digit',
	  hour: '2-digit',
	  minute: '2-digit',
	  second: '2-digit',
	  hour12: true
	});
	pdf.text(`Generated on: ${formattedDate}`, margin, yPos + 5);
	yPos += 10;
	pdf.text("Generated via GarmentIQ.ly.gd.edu.kg - No liability assumed.", margin, yPos + 5);
	yPos += 15;

	const vb = clone.viewBox.baseVal;
	const ratio = vb.width / vb.height;
	const maxImgH = 80;
	let imgH = maxImgH,
		imgW = ratio * imgH;
	const availW = pageW - margin * 2;
	if (imgW > availW) {
		imgW = availW;
		imgH = imgW / ratio;
	}

	const scale = 5;
	const canvas = document.createElement('canvas');
	canvas.width = vb.width * scale;
	canvas.height = vb.height * scale;
	const ctx = canvas.getContext('2d');
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	const renderSVG = async () => {
		if (window.canvg && window.canvg.Canvg) {
			const v = await window.canvg.Canvg.fromString(ctx, svgString, {
				ignoreMouse: true,
				ignoreAnimation: true,
				scaleWidth: canvas.width,
				scaleHeight: canvas.height
			});
			await v.render();
		} else {
			const blob = new Blob([svgString], {
				type: 'image/svg+xml;charset=utf-8'
			});
			const url = URL.createObjectURL(blob);
			await new Promise((res, rej) => {
				const img = new Image();
				img.onload = () => {
					ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
					URL.revokeObjectURL(url);
					res();
				};
				img.onerror = e => {
					URL.revokeObjectURL(url);
					console.warn("SVG→Image fallback failed", e);
					res();
				};
				img.src = url;
			});
		}
	};

	await renderSVG();

	const imgData = canvas.toDataURL('image/png');
	pdf.addImage(imgData, 'PNG', margin, yPos, imgW, imgH);
	yPos += imgH + 10;

	pdf.setFontSize(10);
	const cols = {
		start: 15,
		end: 15,
		name: 40,
		desc: pageW - margin * 2 - (15 + 15 + 40)
	};
	pdf.text('Start', margin, yPos);
	pdf.text('End', margin + cols.start, yPos);
	pdf.text('Name', margin + cols.start + cols.end, yPos);
	pdf.text('Description', margin + cols.start + cols.end + cols.name, yPos);
	yPos += 5;

	const table = document.getElementById('garmentMeasurementDetailTable');
	for (let i = 1; i < table.rows.length; i++) {
		const row = table.rows[i];
		const chk = row.cells[0].querySelector('input[type="checkbox"]');
		if (!chk || !chk.checked) continue;

		const start = row.cells[1].textContent.trim();
		const end = row.cells[2].textContent.trim();
		const name = row.cells[3].querySelector('input').value.trim();
		const desc = row.cells[4].querySelector('input').value.trim();

		if (!name || !desc) {
			alert('Please fill out the Name and Description for all checked measurements.');
			return;
		}

		pdf.text(start, margin, yPos);
		pdf.text(end, margin + cols.start, yPos);
		pdf.text(name, margin + cols.start + cols.end, yPos);

		const wrapped = pdf.splitTextToSize(desc, cols.desc);
		pdf.text(wrapped, margin + cols.start + cols.end + cols.name, yPos);
		yPos += wrapped.length * 4.25;

		if (yPos > pageH - margin) {
			pdf.addPage();
			yPos = margin;
		}
	}

	pdf.save(`${garment}.pdf`);
}
