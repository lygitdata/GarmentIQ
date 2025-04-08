const canvas = document.getElementById("landmarkCanvas");
const ctx = canvas.getContext("2d");
const imageInput = document.getElementById("imageInput");
const dragStatus = document.getElementById("dragStatus");
const removeStatus = document.getElementById("removeStatus");
const dragModeButton = document.getElementById("dragModeButton");
const removeModeButton = document.getElementById("removeModeButton");

let image = new Image();
let imageFileName = "image.png"; // Default filename
let originalWidth = 0, originalHeight = 0;
let scale = 1;
// offsetX/Y store the top-left corner of the image relative to the canvas top-left, in canvas pixels
let offsetX = 0, offsetY = 0;
let landmarks = []; // Stores { id, x, y } where x,y are in ORIGINAL image coordinates
let nextId = 1;

let dragMode = false;
let removeMode = false;
let isDragging = false;
let dragStartX = 0, dragStartY = 0;
let initialOffsetX = 0, initialOffsetY = 0;

// --- Initialization ---
function resizeCanvas() {
    // Set canvas internal buffer size based on its CSS dimensions
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight; // Use the fixed height
     // Redraw if image exists after resize
     if (image.src) {
        // Recalculate initial fit if needed or just redraw
        if (originalWidth > 0) {
            calculateInitialFit(); // Recalculate centering if width changed
            drawImageAndLandmarks();
        }
    } else {
         clearCanvas(); // Clear if no image
    }
}

// Call resizeCanvas initially and whenever the window resizes
window.addEventListener('resize', resizeCanvas);
// Initialize canvas size right away
resizeCanvas();

 function clearCanvas() {
    ctx.fillStyle = '#f0f0f0'; // Match background color
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// --- Image Loading ---
imageInput.addEventListener("change", function(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image/")) {
        imageFileName = file.name;
        const reader = new FileReader();
        reader.onload = function(event) {
            image.onload = () => {
                originalWidth = image.naturalWidth;
                originalHeight = image.naturalHeight;
                resetLandmarks(); // Clear landmarks from previous image
                calculateInitialFit();
                drawImageAndLandmarks();
            };
            image.onerror = () => {
                alert("Error loading image.");
                 clearCanvas();
            };
            image.src = event.target.result;
        };
        reader.readAsDataURL(file);
    } else {
        alert("Please select a valid image file (jpg, jpeg, png).");
         // Optionally clear canvas if invalid file selected
         // image.src = ""; // Clear image object
         // clearCanvas();
         // resetLandmarks();
    }
});

function calculateInitialFit() {
     if (!originalHeight || !canvas.height) return; // Ensure we have dimensions

    scale = canvas.height / originalHeight; // Fit height
    // Center horizontally
    offsetX = (canvas.width - (originalWidth * scale)) / 2;
    offsetY = 0; // Align top
}

// --- Drawing ---
function drawImageAndLandmarks() {
    if (!image.src || !originalWidth) {
        clearCanvas(); // Don't draw if no image
         return;
    }

    clearCanvas();

    ctx.save(); // Save context state (transformations)
    // Apply translation and scaling FROM the canvas origin
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    // Draw the image at (0,0) in the transformed coordinates
    ctx.drawImage(image, 0, 0, originalWidth, originalHeight);

    // Draw landmarks (coordinates are in original image space)
    landmarks.forEach(lm => {
        const landmarkRadiusScreen = 10; // Fixed pixel size on screen
        const landmarkRadiusImage = landmarkRadiusScreen / scale; // Size relative to image

        // Draw Circle
        ctx.beginPath();
        ctx.arc(lm.x, lm.y, landmarkRadiusImage, 0, Math.PI * 2);
        ctx.fillStyle = "green";
        ctx.fill();

        // Draw ID Text
        ctx.fillStyle = "white";
        // Adjust font size based on scale, but keep it readable
        const fontSize = Math.max(12, Math.round(14 / scale)); // Ensure min size, scale down otherwise
         ctx.font = `${fontSize}px Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(lm.id.toString(), lm.x, lm.y);

        // Highlight for removal
         if (removeMode) {
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2 / scale; // Line width relative to image scale
            ctx.stroke(); // Draw the red circle outline over the green
        }
    });

    ctx.restore(); // Restore context state
}

// --- Coordinate Transformation ---
function getMousePos(evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left, // Mouse X relative to canvas
        y: evt.clientY - rect.top   // Mouse Y relative to canvas
    };
}

function screenToImageCoords(screenX, screenY) {
    return {
        x: (screenX - offsetX) / scale,
        y: (screenY - offsetY) / scale
    };
}

// --- Event Handlers ---
canvas.addEventListener("mousedown", function(e) {
    if (!image.src || !originalWidth) return; // Need image loaded

    if (dragMode) {
        isDragging = true;
        const pos = getMousePos(e);
        dragStartX = pos.x;
        dragStartY = pos.y;
        initialOffsetX = offsetX;
        initialOffsetY = offsetY;
        canvas.style.cursor = "grabbing";
    }
});

canvas.addEventListener("mousemove", function(e) {
     if (!image.src || !originalWidth) return;

    if (isDragging && dragMode) {
        const pos = getMousePos(e);
        const dx = pos.x - dragStartX;
        const dy = pos.y - dragStartY;
        offsetX = initialOffsetX + dx;
        offsetY = initialOffsetY + dy;
        drawImageAndLandmarks();
    }
});

canvas.addEventListener("mouseup", function(e) {
    if (!image.src || !originalWidth) return;

    if (isDragging && dragMode) {
        isDragging = false;
        canvas.style.cursor = "grab";
    }
    // No else if: click happens on mouseup
});

canvas.addEventListener("mouseleave", function(e) {
     if (isDragging && dragMode) {
         isDragging = false;
         canvas.style.cursor = "grab";
     }
});

canvas.addEventListener("click", function(e) {
     if (!image.src || !originalWidth || isDragging) return; // Don't place point if dragging just finished

    const pos = getMousePos(e);
    const imageCoords = screenToImageCoords(pos.x, pos.y);

    if (removeMode) {
        // Find landmark near click (in image coordinates)
        const clickRadiusImage = 15 / scale; // Click target radius
        let removed = false;
        for (let i = landmarks.length - 1; i >= 0; i--) {
            const lm = landmarks[i];
            const dx = imageCoords.x - lm.x;
            const dy = imageCoords.y - lm.y;
            if (dx * dx + dy * dy < clickRadiusImage * clickRadiusImage) {
                removeLandmarkByIndex(i);
                removed = true;
                break; // Remove only one
            }
        }
        if (removed) {
            drawImageAndLandmarks();
        } else {
            // Optional: give feedback if click didn't hit anything in remove mode
            console.log("Click did not hit a landmark in remove mode.");
        }

    } else if (!dragMode) { // Only add if not in drag mode or remove mode
        addLandmark(imageCoords.x, imageCoords.y);
        drawImageAndLandmarks();
    }
});

 // Add wheel event listener for zooming
 canvas.addEventListener('wheel', function(e) {
     if (!image.src || !originalWidth) return;
     e.preventDefault(); // Prevent page scrolling

     const factor = e.deltaY < 0 ? 1.1 : 0.9; // Zoom in or out
     const pos = getMousePos(e);
     zoom(factor, pos.x, pos.y); // Zoom towards mouse cursor
 }, { passive: false }); // Need passive: false to preventDefault


// --- Landmark Management ---
function addLandmark(x, y) {
    // Check bounds (optional: prevent adding outside image)
    // if (x < 0 || x > originalWidth || y < 0 || y > originalHeight) return;

    landmarks.push({ id: nextId, x: x, y: y });
    nextId++;
}

function removeLandmarkByIndex(index) {
    if (index < 0 || index >= landmarks.length) return;

    landmarks.splice(index, 1); // Remove the landmark at the found index

    // Renumber subsequent landmarks
    nextId = 1;
    landmarks.forEach(lm => {
        lm.id = nextId++;
    });
}

function resetLandmarks() {
    landmarks = [];
    nextId = 1;
     if (image.src && originalWidth) { // Only redraw if image exists
         drawImageAndLandmarks();
     }
}

// --- UI Controls ---
function zoom(factor, zoomCenterX = canvas.width / 2, zoomCenterY = canvas.height / 2) {
    if (!image.src || !originalWidth) return;

     // Calculate image coordinates under the mouse cursor BEFORE zoom
     const imageXBeforeZoom = (zoomCenterX - offsetX) / scale;
     const imageYBeforeZoom = (zoomCenterY - offsetY) / scale;

    // Apply zoom factor
    scale *= factor;

     // Calculate new offset to keep the point under the cursor stationary
     offsetX = zoomCenterX - (imageXBeforeZoom * scale);
     offsetY = zoomCenterY - (imageYBeforeZoom * scale);


    drawImageAndLandmarks();
}

function toggleDragMode() {
    dragMode = !dragMode;
    dragStatus.textContent = dragMode ? "On" : "Off";
    dragModeButton.classList.toggle("active", dragMode);

    if (dragMode) {
        removeMode = false; // Turn off remove mode if drag mode is activated
        removeStatus.textContent = "Off";
        removeModeButton.classList.remove("active");
        canvas.style.cursor = "grab";
        drawImageAndLandmarks(); // Redraw to remove remove-highlight if necessary
    } else {
        canvas.style.cursor = "crosshair"; // Default cursor
    }
}

function toggleRemoveMode() {
    removeMode = !removeMode;
    removeStatus.textContent = removeMode ? "On" : "Off";
    removeModeButton.classList.toggle("active", removeMode);

    if (removeMode) {
        dragMode = false; // Turn off drag mode
        dragStatus.textContent = "Off";
        dragModeButton.classList.remove("active");
        isDragging = false; // Ensure dragging stops if it was active
        canvas.style.cursor = "pointer"; // Indicate clickable points
    } else {
        canvas.style.cursor = dragMode ? "grab" : "crosshair"; // Revert based on dragMode
    }
    drawImageAndLandmarks(); // Redraw to show/hide remove highlights
}

// --- Exporting ---
function exportImage() {
    if (!image.src || !originalWidth) {
        alert("Please load an image first.");
        return;
    }
    if (landmarks.length === 0) {
        alert("No landmarks to export.");
        return;
    }

    // Create a temporary canvas scaled to the original image size
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = originalWidth;
    exportCanvas.height = originalHeight;
    const exportCtx = exportCanvas.getContext("2d");

    // Draw the original image
    exportCtx.drawImage(image, 0, 0, originalWidth, originalHeight);

    // Draw landmarks onto the export canvas (using original coordinates)
     const landmarkRadiusOriginal = 10; // Radius in original pixels
     const fontSizeOriginal = 12; // Font size in original pixels

    landmarks.forEach(lm => {
         // Draw Circle
         exportCtx.beginPath();
         exportCtx.arc(lm.x, lm.y, landmarkRadiusOriginal, 0, Math.PI * 2);
         exportCtx.fillStyle = "green";
         exportCtx.fill();

         // Draw ID Text
         exportCtx.fillStyle = "white";
         exportCtx.font = `${fontSizeOriginal}px Arial`;
         exportCtx.textAlign = "center";
         exportCtx.textBaseline = "middle";
         exportCtx.fillText(lm.id.toString(), lm.x, lm.y);
    });

    // Create download link
    const link = document.createElement("a");
    link.download = "anno_" + imageFileName; // Add "anno_" prefix
    link.href = exportCanvas.toDataURL("image/png"); // Use PNG for export
    link.click();
}

function exportMetadata() {
     if (!image.src || !originalWidth) {
         alert("Please load an image first.");
         return;
     }
    if (landmarks.length === 0) {
        alert("No landmarks to export.");
        return;
    }

    const metadata = landmarks.map(lm => {
        const distances = {};
        landmarks.forEach(otherLm => {
            if (lm.id !== otherLm.id) {
                const dx = lm.x - otherLm.x;
                const dy = lm.y - otherLm.y;
                // Calculate distance in original image pixel units
                distances[otherLm.id] = Math.sqrt(dx * dx + dy * dy);
            }
        });
        return {
            id: lm.id,
            x: lm.x, // Already in original image coordinates
            y: lm.y, // Already in original image coordinates
            distances: distances
        };
    });

    const dataStr = JSON.stringify(metadata, null, 2); // Pretty print JSON
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    // Get filename without extension and add .json
    let baseName = imageFileName.includes('.') ? imageFileName.substring(0, imageFileName.lastIndexOf('.')) : imageFileName;
    link.download = baseName + ".json";
    link.href = url;
    link.click();

    // Clean up the object URL
    URL.revokeObjectURL(url);
}