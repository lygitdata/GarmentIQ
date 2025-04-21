// --- DOM Element References ---
const garmentContainer = document.getElementById('garmentDisplayContainer');
const dragModeButton = document.getElementById('dragModeButton');
const dragStatus = document.getElementById('dragStatus');
const zoomInButton = document.getElementById('zoomInButton');
const zoomOutButton = document.getElementById('zoomOutButton');
const garmentSelector = document.getElementById('garmentSelector');

// --- State Variables ---
const selectedPoints = new Map(); // Stores selected landmark IDs per garment { svgId: Set(landmarkId) }

// Zoom/Pan State
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let initialOffsetX = 0;
let initialOffsetY = 0;
let dragMode = false;
let justFinishedDrag = false; // Flag to prevent click immediately after drag

// --- Constants ---
const BUTTON_ZOOM_FACTOR = 1.2; // Zoom factor for button clicks & scroll
const MIN_SCALE = 0.2;        // Minimum zoom scale allowed
const MAX_SCALE = 10;         // Maximum zoom scale allowed

// --- Helper Functions ---

/**
 * Gets mouse coordinates relative to a container element.
 * @param {MouseEvent} evt - The mouse event.
 * @param {HTMLElement} container - The container element.
 * @returns {{x: number, y: number}} Coordinates relative to the container.
 */
function getMousePos(evt, container) {
    const rect = container.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

/**
 * Finds the currently visible SVG element within the container.
 * @returns {SVGElement | null} The active SVG element or null if none is found.
 */
function getActiveSvg() {
    return garmentContainer.querySelector('svg.show');
}

/**
 * Applies the current scale and offset transformations to the active SVG.
 */
function applyTransform() {
    const activeSvg = getActiveSvg();
    if (activeSvg) {
        activeSvg.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
    }
}

/**
 * Converts screen coordinates (relative to the container) into the SVG's internal coordinate system.
 * Assumes SVG transform-origin is (0, 0).
 * @param {number} screenX - X coordinate relative to the container.
 * @param {number} screenY - Y coordinate relative to the container.
 * @returns {{x: number, y: number}} Coordinates in the SVG's internal space.
 */
function screenToSvgCoords(screenX, screenY) {
    return {
        x: (screenX - offsetX) / scale,
        y: (screenY - offsetY) / scale
    };
}

/**
 * Updates the text, class, and disabled state of control buttons (Drag mode, Zoom).
 * Also updates the cursor style for the container.
 */
function updateButtonStates() {
    // Drag Mode Button
    dragStatus.textContent = dragMode ? "On" : "Off";
    dragModeButton.classList.toggle("active", dragMode);

    // Container Cursor
    if (dragMode) {
        garmentContainer.style.cursor = isDragging ? "grabbing" : "grab";
    } else {
        // Use default cursor, letting landmarks handle their pointer if needed
        garmentContainer.style.cursor = "default";
    }

    // Zoom Buttons (Disable at limits)
    zoomInButton.disabled = scale >= MAX_SCALE;
    zoomOutButton.disabled = scale <= MIN_SCALE;
}

// --- Core Application Logic ---

/**
 * Hides the currently visible SVG, shows the selected one, resets view,
 * and restores landmark selections for the newly shown SVG.
 */
function showGarment() {
    const selectedGarmentId = garmentSelector.value;

    // Hide and reset transform on the previously shown SVG
    const previousSvg = getActiveSvg();
    if (previousSvg) {
        previousSvg.style.transform = 'none'; // Reset transform explicitly
        previousSvg.classList.remove('show');
        previousSvg.style.display = 'none'; // Ensure it's hidden
    }

    // Reset landmark selection UI (circles)
    document.querySelectorAll('#garmentDisplayContainer .point').forEach(c => c.classList.remove('selected'));

    // Reset zoom/pan state for the new view
    scale = 1;
    offsetX = 0;
    offsetY = 0;
    isDragging = false; // Ensure dragging stops
    justFinishedDrag = false;

    // Apply default transform (effectively none) and update ALL button states
    applyTransform(); // Needed even if no SVG is shown yet, to set initial state for buttons
    updateButtonStates();

    // Show the selected SVG and restore its selections
    if (selectedGarmentId) {
        const selectedSvg = document.getElementById(selectedGarmentId);
        if (selectedSvg) {
            selectedSvg.style.display = 'block'; // Make it visible first
            // Apply the freshly reset transform before adding 'show' class
            selectedSvg.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
            // Use setTimeout to allow display:block to render before adding class for transition
            setTimeout(() => {
                selectedSvg.classList.add('show') // Add class to trigger transition/make active
            }, 10);

            // Restore selections for this garment from the map
            const currentSelections = selectedPoints.get(selectedGarmentId);
            if (currentSelections) {
                currentSelections.forEach(id => {
                    // Query within the specific SVG for safety
                    const circle = selectedSvg.querySelector(`.landmark[data-id="${id}"] circle.point`);
                    if (circle) {
                        circle.classList.add('selected');
                    }
                });
            }
        } else {
            console.warn(`SVG element with ID "${selectedGarmentId}" not found.`);
        }
    }
}


/**
 * Generates and triggers download of a JSON file containing the selected landmark IDs and descriptions.
 */
function generateJsonBlob() {
    const garment = garmentSelector.value;
    if (!garment) {
        alert("Please choose a garment first.");
        return;
    };

    const output = {};
    output[garment] = {}; // Use garment ID as the key

    const svg = document.getElementById(garment);
    if (!svg) {
        console.error(`Cannot find SVG for garment: ${garment}`);
        return;
    }

    const currentSelections = selectedPoints.get(garment);
    if (!currentSelections || currentSelections.size === 0) {
        alert("Please select at least one landmark before exporting JSON.");
        return;
    }

    // Populate the output object with selected landmark descriptions and IDs
    svg.querySelectorAll('.landmark').forEach(group => {
        const id = parseInt(group.dataset.id);
        const desc = group.dataset.description;
        if (currentSelections.has(id)) {
            output[garment][desc] = id; // Map description to ID
        }
    });


    const blob = new Blob([JSON.stringify(output, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    // Sanitize filename slightly
    const safeGarmentName = garment.replace(/[^a-z0-9]/gi, '_').toLowerCase();
    link.download = `${safeGarmentName}_measurement_schema.json`;
    document.body.appendChild(link); // Required for Firefox
    link.click();
    document.body.removeChild(link); // Clean up link element
    setTimeout(() => URL.revokeObjectURL(url), 100); // Clean up object URL after download starts
}

/**
 * Generates and triggers download of a PDF file showing the SVG garment
 * with selected landmarks highlighted and listed.
 * Requires jsPDF library and optionally canvg for better SVG rendering.
 */
async function exportToPDF() {
    const garment = garmentSelector.value;
    if (!garment) {
         alert("Please choose a garment first.");
         return;
    }

    const svg = document.getElementById(garment);
    if (!svg) {
        console.error(`Cannot find SVG for garment: ${garment}`);
        return;
    }

    // Check if any landmark is selected
    const selections = selectedPoints.get(garment);
    if (!selections || selections.size === 0) {
        alert("Please select at least one landmark before exporting PDF.");
        return;
    }

    // --- PDF Generation ---
    // Create a clone WITHOUT the current view transform for export
    const clone = svg.cloneNode(true);
    clone.style.display = "block"; // Ensure it's displayable
    clone.style.transform = 'none'; // Remove view transforms
    clone.removeAttribute('class'); // Remove 'show' class if present
    // Ensure selected class is on the clone for rendering
    selections.forEach(id => {
         const circle = clone.querySelector(`.landmark[data-id="${id}"] circle.point`);
         if (circle) circle.classList.add('selected');
    });

    const svgString = new XMLSerializer().serializeToString(clone);

    // --- jsPDF Initialization ---
    // Ensure jsPDF is loaded (assuming it's included via <script>)
    if (typeof window.jspdf === 'undefined' || typeof window.jspdf.jsPDF === 'undefined') {
        alert("jsPDF library is not loaded. Cannot export to PDF.");
        console.error("jsPDF not found. Make sure it's included in your HTML.");
        return;
    }
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF(); // Default: Portrait, mm, A4

    // --- PDF Content ---
    const capitalizedGarment = garment.charAt(0).toUpperCase() + garment.slice(1).replace(/_/g, ' ');
    const title = `${capitalizedGarment} Measurement Schema`;
    const margin = 15; // Page margin in mm
    const availableWidth = pdf.internal.pageSize.getWidth() - 2 * margin;

    pdf.setFontSize(16);
    pdf.text(title, margin, margin + 5); // Add title with margin

    // --- Render SVG to PDF ---
    // Use Canvas rendering for better reliability/compatibility
    try {
        const canvas = document.createElement("canvas");
        const svgWidth = svg.viewBox.baseVal.width;
        const svgHeight = svg.viewBox.baseVal.height;
        const ratio = svgWidth / svgHeight

        // Calculate image dimensions in PDF (mm)
        const pdfImageWidth = 100; 
        const pdfImageHeight = 100 / ratio;

        // Render canvas larger for better resolution
        const renderScale = 8;
        canvas.width = pdfImageWidth * renderScale;
        canvas.height = pdfImageHeight * renderScale;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = 'white'; // Set background
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Use canvg if available (recommended: include it via <script>)
        if (typeof canvg !== 'undefined' && typeof canvg.Canvg !== 'undefined') {
             const v = await canvg.Canvg.fromString(ctx, svgString, {
                 ignoreMouse: true,
                 ignoreAnimation: true,
                 ignoreDimensions: true, // Let canvg determine size based on SVG content/viewBox
                 scaleWidth: canvas.width,
                 scaleHeight: canvas.height,
                 offsetX: 0,
                 offsetY: 0
             });
             await v.render();
             console.log("SVG rendered with Canvg.");
        } else {
             // Basic fallback: Render SVG string to Image, then draw Image to Canvas
            console.warn("Canvg not found. Using basic SVG rendering for PDF (may have issues).");
            const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
            const url = URL.createObjectURL(svgBlob);
            const img = new Image();

            await new Promise((resolve, reject) => {
                img.onload = () => {
                     ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                     URL.revokeObjectURL(url);
                     resolve();
                 };
                 img.onerror = (err) => {
                     URL.revokeObjectURL(url);
                     console.error("Failed to load SVG blob into image for PDF.", err);
                     reject(new Error("Failed to load SVG blob into image."));
                 };
                 img.src = url;
            });
        }

        const imgData = canvas.toDataURL("image/png"); // Get image data from canvas
        pdf.addImage(imgData, "PNG", margin, margin + 15, pdfImageWidth, pdfImageHeight); // Add image below title

    } catch (error) {
        console.error("Error rendering SVG to canvas for PDF:", error);
        alert("An error occurred while preparing the image for the PDF. Check console.");
        // Optionally, still try to save PDF without image? Or just return.
        return;
    }
    // --- End SVG Rendering ---

    // --- Add List of Selected Points ---
    let y = pdf.internal.pageSize.getHeight() + margin * 2; // Start Y below the image + spacing
    const maxY = pdf.internal.pageSize.getHeight() - margin; // Max Y for content on page

    y += 7; // Line height + spacing

    pdf.setFontSize(10);
    let pointCount = 0;
    svg.querySelectorAll('.landmark').forEach(group => {
        const id = parseInt(group.dataset.id);
        const desc = group.dataset.description;
        if (selections.has(id)) {
            // Check if adding text exceeds page height
            if (y > maxY) {
                 pdf.addPage();
                 y = margin; // Reset y for new page (start from top margin)
                 pdf.setFontSize(10); // Reset font size for new page
            }
            pdf.text(`  - ${id}: ${desc}`, margin, y); // Format: ID: Description
            y += 5; // Smaller line height for list
            pointCount++;
        }
    });

    if (pointCount === 0) { // Should be caught earlier, but double check
        alert("No landmarks were selected for the PDF export.");
        return;
    }

    // --- Save PDF ---
    const safeGarmentName = garment.replace(/[^a-z0-9]/gi, '_').toLowerCase();
    pdf.save(`${safeGarmentName}_measurement_schema.pdf`);
}


/**
 * Selects all available landmarks on the currently shown SVG.
 */
function selectAllLandmarks() {
    const garment = garmentSelector.value;
    if (!garment) return;

    const svg = document.getElementById(garment);
    if (!svg) return;

    // Ensure a set exists for this garment
    if (!selectedPoints.has(garment)) {
        selectedPoints.set(garment, new Set());
    }
    const currentSelections = selectedPoints.get(garment);

    // Iterate through landmarks in the SVG
    svg.querySelectorAll('.landmark').forEach(group => {
        const id = parseInt(group.dataset.id);
        const circle = group.querySelector('circle.point');
        // Add to set and update style if not already selected
        if (!currentSelections.has(id)) {
            currentSelections.add(id);
            if (circle) circle.classList.add('selected');
        }
    });
}

/**
 * Deselects all landmarks on the currently shown SVG.
 */
function resetLandmarks() {
    const garment = garmentSelector.value;
    if (!garment) return;

    const svg = document.getElementById(garment);
     if (!svg) return;

    const currentSelections = selectedPoints.get(garment);
    if (currentSelections) {
        // Get all landmark IDs that are currently selected
        const idsToRemove = Array.from(currentSelections);

        idsToRemove.forEach(id => {
            const circle = svg.querySelector(`.landmark[data-id="${id}"] circle.point`);
            if (circle) {
                circle.classList.remove('selected');
            }
            currentSelections.delete(id); // Remove from the Set
        });
    }
}

/**
 * Zooms the view by a given factor, centered on specific screen coordinates.
 * @param {number} factor - Zoom factor (e.g., 1.2 for zoom in, 1/1.2 for zoom out).
 * @param {number} zoomCenterX - X coordinate for the zoom center (relative to container).
 * @param {number} zoomCenterY - Y coordinate for the zoom center (relative to container).
 */
function zoom(factor, zoomCenterX, zoomCenterY) {
    const activeSvg = getActiveSvg();
    if (!activeSvg) return;

    const newScale = scale * factor;

    // Check limits BEFORE applying
    if (newScale < MIN_SCALE || newScale > MAX_SCALE) {
        // Optionally clamp scale here if desired
        // scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));
        updateButtonStates(); // Update buttons even if zoom didn't change
        return; // Don't zoom beyond limits
    }

    // Calculate SVG coordinates under the zoom center BEFORE zoom
    const svgCoordsBeforeZoom = screenToSvgCoords(zoomCenterX, zoomCenterY);

    // Apply zoom factor
    scale = newScale;

    // Calculate new offset to keep the point under the cursor stationary
    offsetX = zoomCenterX - (svgCoordsBeforeZoom.x * scale);
    offsetY = zoomCenterY - (svgCoordsBeforeZoom.y * scale);

    applyTransform();
    updateButtonStates(); // Update zoom button disabled status
}

/**
 * Toggles the drag mode on/off.
 */
function toggleDragMode() {
    dragMode = !dragMode;
    isDragging = false; // Ensure dragging stops if mode changes while dragging
    updateButtonStates();
    // No redraw needed, just cursor/state update
}


// --- Event Listeners Setup ---

// Listener for Garment Selection Change
garmentSelector.addEventListener('change', showGarment);

// Listeners for Landmark Clicks (using event delegation on the container)
garmentContainer.addEventListener('click', (e) => {
    // Check if the click target is a landmark circle or inside a landmark group
    const landmarkGroup = e.target.closest('.landmark');
    if (!landmarkGroup) return; // Click wasn't on or inside a landmark group

    // Ensure it's the correct circle if target is inside group but not circle itself
    const circle = landmarkGroup.querySelector('circle.point');
    if (!circle || !landmarkGroup.contains(e.target)) return; // Ensure click is relevant

    const id = parseInt(landmarkGroup.dataset.id);
    const svg = landmarkGroup.closest('svg');
    if (!svg) return;
    const svgId = svg.id;

    // Prevent selection if dragging just finished or if drag mode is on
    if (justFinishedDrag || dragMode) {
        // console.log("Selection skipped due to drag/mode");
        return; // Do nothing
    }

    // Add/Remove from selection
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


// Listeners for Drag/Pan on the Container
garmentContainer.addEventListener("mousedown", function(e) {
    const activeSvg = getActiveSvg();
    if (!activeSvg || !dragMode) return; // Only drag if mode is on

    // Prevent initiating drag if the click target is specifically a landmark circle
    if (e.target.classList.contains('point') || e.target.closest('.landmark')) {
        // console.log("Drag prevented: click on landmark.");
        return;
    }

    isDragging = true;
    justFinishedDrag = false;
    const pos = getMousePos(e, garmentContainer);
    dragStartX = pos.x;
    dragStartY = pos.y;
    initialOffsetX = offsetX;
    initialOffsetY = offsetY;
    updateButtonStates(); // Updates cursor to grabbing
    e.preventDefault(); // Prevent text selection, etc.
});

garmentContainer.addEventListener("mousemove", function(e) {
    if (!isDragging || !dragMode) return;
    const activeSvg = getActiveSvg();
    if (!activeSvg) { isDragging = false; updateButtonStates(); return; } // Safety check

    const pos = getMousePos(e, garmentContainer);
    const dx = pos.x - dragStartX;
    const dy = pos.y - dragStartY;
    offsetX = initialOffsetX + dx;
    offsetY = initialOffsetY + dy;
    applyTransform();
    e.preventDefault(); // Prevent issues during drag
});

garmentContainer.addEventListener("mouseup", function(e) {
    if (!isDragging || !dragMode) return;

    const pos = getMousePos(e, garmentContainer);
    const dx = pos.x - dragStartX;
    const dy = pos.y - dragStartY;
    // Set flag if mouse moved significantly
    justFinishedDrag = (Math.abs(dx) > 3 || Math.abs(dy) > 3);

    isDragging = false;
    updateButtonStates(); // Updates cursor back to grab
    e.preventDefault();

    // Reset the flag shortly after mouseup allows click event (if any) to check it
    if (justFinishedDrag) {
        setTimeout(() => { justFinishedDrag = false; }, 50);
    }
});

garmentContainer.addEventListener("mouseleave", function(e) {
    if (isDragging && dragMode) { // Only act if dragging was in progress
        isDragging = false;
        justFinishedDrag = true; // Leaving container ends the drag action
        updateButtonStates();
        // Reset flag shortly after
        setTimeout(() => { justFinishedDrag = false; }, 50);
    }
});

// Listener for Scroll Wheel Zoom on the Container
garmentContainer.addEventListener('wheel', function(e) {
    const activeSvg = getActiveSvg();
    if (!activeSvg) return; // Don't zoom if no SVG is visible

    e.preventDefault(); // Prevent page scrolling

    const factor = e.deltaY < 0 ? BUTTON_ZOOM_FACTOR : 1 / BUTTON_ZOOM_FACTOR;
    const pos = getMousePos(e, garmentContainer);
    zoom(factor, pos.x, pos.y);
}, { passive: false }); // Need passive: false to preventDefault

// Listeners for Zoom Buttons
zoomInButton.addEventListener('click', function() {
    const activeSvg = getActiveSvg();
    if (!activeSvg) return;
    const centerX = garmentContainer.clientWidth / 2;
    const centerY = garmentContainer.clientHeight / 2;
    zoom(BUTTON_ZOOM_FACTOR, centerX, centerY);
});

zoomOutButton.addEventListener('click', function() {
    const activeSvg = getActiveSvg();
    if (!activeSvg) return;
    const centerX = garmentContainer.clientWidth / 2;
    const centerY = garmentContainer.clientHeight / 2;
    zoom(1 / BUTTON_ZOOM_FACTOR, centerX, centerY);
});

// Listener for Drag Mode Button
dragModeButton.addEventListener('click', toggleDragMode);

// --- Initial Page Load Setup ---
document.addEventListener('DOMContentLoaded', () => {
    showGarment(); // Initial call to set up based on default dropdown value
});
