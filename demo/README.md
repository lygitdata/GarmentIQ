# GarmentIQ local demo

A local Flask server serves the frontend and runs the Python package in this
checkout. No Colab, ngrok, Node build or remote inference server is required.

## Start

Run from the repository root. Python 3.11 or newer is required (3.12 recommended).

```sh
uv venv --python 3.12 .venv
uv pip install -e ".[web]"
.venv/bin/python -m demo.download_models
.venv/bin/python -m demo.app --reload
```

Open **http://127.0.0.1:5002**. If the environment and weights already exist, only
the last command is needed. The first measurement loads the networks; later
measurements reuse them. CPU is the default. Request an accelerator explicitly:

```sh
.venv/bin/python -m demo.app --device mps --reload
# NVIDIA systems:
.venv/bin/python -m demo.app --device cuda --reload
```

Without uv, create a virtual environment with `python3.12 -m venv .venv`, then
use `.venv/bin/python -m pip install -e ".[web]"`.

## Make changes

| File | Purpose |
| --- | --- |
| `demo/static/index.html` | Page structure and controls |
| `demo/static/style.css` | Layout, colours and responsive styles |
| `demo/static/app.js` | Uploads, requests and result rendering |
| `demo/static/editor.js` | Category diagrams, derived points and measurement authoring |
| `demo/static/viewer.js` | Interactive landmarks, dependency highlighting and measurements |
| `demo/instructions.py` | Schema validation, revision checks, backups and saving defaults |
| `demo/calibration.py` | Separate horizontal/vertical centimetre calibration |
| `demo/app.py` | Flask routes, upload validation, isolated runs and artifact downloads |
| `demo/pipeline.py` | Model configuration and integration with the Python engine |
| `src/garmentiq/` | Measurement algorithms; installed in editable mode |

Refresh the browser after frontend edits. `--reload` restarts the server after
Python edits; loaded models will be initialized again on the next measurement.

## Measurement behaviour

- Uses TinyViT (Inditex fine-tuned), BiRefNet and HRNet, matching the web Tailor demo.
- Supports the nine bundled garment categories, optional landmark refinement,
  optional landmark derivation and the actual selected background colour.
- Reports Euclidean distances in **pixels**, with optional per-image **centimetre** calibration.
- When derivation is disabled, omits measurements that require derived points
  rather than using the instruction files' illustrative coordinates.
- Shows original images, detected landmarks, masks and background composites when
  available. JSON and CSV files can be downloaded.
- The selected background replaces background pixels **before HRNet detection**.
  Interactive and exported measurement images use that same background.
- Hover or focus a point to inspect its name, pixel/cm coordinates, confidence
  and measurements. Measurements that depend indirectly on a point through a
  derivation rule are highlighted too. Click a point to pin its information.
- Matting and SAM are available in the Python library but are not exposed in this
  initial local interface.

The CSV download is a flattened measurement table with pixel distances and,
when calibrated, centimetre distances. The original artifact metadata remains
available in each run's `output/metadata.csv`. Individual JSON files include
landmark names, derived rules, image dimensions, calibration and measurements.

## Custom category instructions

Open **Define your measurements**, choose a category, and define
measurements between detected or derived landmarks. Click diagram points to
choose the start and end of a new measurement; existing rows can be edited too.

For a derived landmark, choose its name, direction, placement mode and five detected references:

1. Starting point through which the derived line passes.
2. First point of the direction reference pair.
3. Second point of the direction reference pair; run parallel/perpendicular to 2–3.
4. First point of the intersection reference pair.
5. Second point of that pair; its line identifies the intended contour crossing.

**Exact intersection** keeps the mathematical intersection of the line through
point 1 and the line through 4–5. It does not consult or snap to the mask.
**Nearest edge** follows the line through point 1 to the contour crossing nearest
that intersection. This is the default for existing instructions.

The choice is stored per derived point as `derivation.intersection_mode`, with
values `exact` or `nearest_edge`. Omitting the field preserves the original
nearest-edge behaviour. Diagram coordinates are illustrative;
the pipeline derives coordinates anew for each image. References must be detected
points because HRNet's coordinate array contains only detected landmarks.

Use **Apply point to draft**, add the measurement, then **Save & replace category
defaults**. This overwrites `src/garmentiq/instruction/<category>.json`. The previous
file is saved under `.local-demo/instruction-backups/`. Stale browser revisions
are rejected instead of silently overwriting newer edits. A running measurement
must finish before instructions can be saved.

Subsequent runs read the saved category file without reloading model weights.
Each run snapshots its instructions, available as **Instruction used**. The
category selector controls which schema you edit; classification still chooses
the category for an uploaded garment. The diagrams and base landmark names in
`garments.json` are adapted from the official website's instruction-generation demo.

## Centimetre calibration

Under each selected image, optionally provide the physical width and height of
the **whole image frame**, in the displayed orientation. Leave both blank to keep
pixels only. These values are not the garment's width or height.

With image dimensions `W × H` pixels and physical dimensions `w × h` cm:

```text
sx = w / W; sy = h / H
distance_cm = sqrt((dx * sx)^2 + (dy * sy)^2)
```

This uses independent scales for both axes rather than averaging them. The
calibration assumes a flat measurement plane viewed front-on; it does not correct
perspective or infer physical size from an uncalibrated photograph.

## Local data and API

Ignored local state lives under `.local-demo/`: `models/` holds checkpoints and
`runs/<id>/` holds each upload and its results. Successful runs persist until you
remove them. `--model-dir PATH` reuses existing checkpoints; `--data-dir PATH`
changes the run storage root. Run the downloader with `--model-dir PATH` too when
using a custom checkpoint location.

The server binds to `127.0.0.1`. One inference request runs at a time; another gets
HTTP 409. Uploads are limited to 12 images, 32 MB total and 25 megapixels per image.
Uploads are oriented from EXIF metadata and converted to RGB PNG, so results use
the normalized filenames and coordinates.

| Route | Contract |
| --- | --- |
| `GET /api/health` | Readiness, missing dependencies/checkpoints and device |
| `GET /api/instructions` | List editable garment categories |
| `GET /api/instructions/<category>` | Read current JSON, revision and category diagram |
| `PUT /api/instructions/<category>` | Save JSON `instruction` and its last-read `revision` |
| `POST /api/measure` | Multipart `images`, `derive` / `refine`, optional `background`; `calibrations` is a JSON array of `null` or `{width_cm, height_cm}` in upload order |
| `GET /api/runs/<id>/<path>` | Retrieve a run's image, JSON or CSV artifact |

The API uses file URLs instead of base64 responses. It is specific to this local
interface and does not replace the hosted website's notebook protocol.

## Checks

```sh
.venv/bin/python -m unittest discover -s demo/tests -v
node --check demo/static/app.js
node --check demo/static/editor.js
node --check demo/static/viewer.js
```

The checks cover the API, safe instruction saves, calibration, synthetic contour
derivation, and routing the selected background into detection and output.
They require the installed package but do not load trained weights. Real
measurement requires all three downloaded checkpoints.
