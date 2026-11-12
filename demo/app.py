"""Run with `python -m demo.app` from the repository root."""
import argparse
import json
from io import BytesIO
from pathlib import Path
import re
import shutil
import threading
import uuid

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from .pipeline import LocalPipeline, runtime_status
from .instructions import CATALOG, InstructionStore, RevisionConflict
from .calibration import validate_calibrations

ROOT = Path(__file__).resolve().parent.parent


def create_app(data_dir=None, model_dir=None, device="cpu", runner=None, instruction_dir=None):
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    data_dir = Path(data_dir or ROOT / ".local-demo").resolve()
    model_dir = Path(model_dir or data_dir / "models").resolve()
    runs_dir = data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    store = InstructionStore(data_dir / "instruction-backups", instruction_dir)
    pipeline = runner or LocalPipeline(model_dir, device, store.directory)
    pipeline_lock = threading.Lock()

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(**runtime_status(model_dir), device=device,
                       model_dir=str(model_dir), busy=pipeline_lock.locked())

    @app.get("/api/instructions")
    def categories():
        return jsonify(categories=sorted(CATALOG))

    @app.route("/api/instructions/<category>", methods=["GET", "PUT"])
    def instructions(category):
        if category not in CATALOG:
            return jsonify(error="Unknown garment category."), 404
        if not pipeline_lock.acquire(blocking=False):
            return jsonify(error="Wait for the current measurement to finish before editing instructions."), 409
        try:
            if request.method == "GET":
                return jsonify(store.read(category))
            body = request.get_json()
            if not isinstance(body, dict):
                raise ValueError("Supply the category instruction and its revision as JSON.")
            return jsonify(store.save(category, body.get("instruction"), body.get("revision")))
        except RevisionConflict as error:
            return jsonify(error=str(error)), 409
        except ValueError as error:
            return jsonify(error=str(error)), 400
        finally:
            pipeline_lock.release()

    @app.post("/api/measure")
    def measure():
        files = request.files.getlist("images")
        if not files or len(files) > 12:
            return jsonify(error="Choose between 1 and 12 garment images."), 400
        options = {}
        for name in ("derive", "refine"):
            value = request.form.get(name, "true")
            if value not in ("true", "false"):
                return jsonify(error=f"Invalid {name} option."), 400
            options[name] = value == "true"
        color = request.form.get("background", "")
        if color and not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            return jsonify(error="Background must be a six-digit hex colour."), 400
        options["background"] = [int(color[i:i + 2], 16) for i in (1, 3, 5)] if color else None
        if "calibrations" in request.form:
            try:
                options["calibrations"] = validate_calibrations(json.loads(request.form["calibrations"]), len(files))
            except (ValueError, TypeError):
                return jsonify(error="Provide both positive image width and height in cm for each calibrated image, or leave both blank."), 400
        if runner is None:
            status = runtime_status(model_dir)
            if not status["ready"]:
                return jsonify(error="The measurement runtime needs setup. See the setup panel.", **status), 503
        if not pipeline_lock.acquire(blocking=False):
            return jsonify(error="A measurement is already running. Try again when it finishes."), 409
        run_id = uuid.uuid4().hex
        run_dir = runs_dir / run_id
        input_dir = run_dir / "input"
        try:
            input_dir.mkdir(parents=True)
            # Re-encode uploads and use generated prefixes to avoid path/filename collisions.
            for index, upload in enumerate(files):
                try:
                    with Image.open(BytesIO(upload.read())) as image:
                        if image.width * image.height > 25_000_000:
                            raise ValueError("Images must be at most 25 megapixels.")
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        stem = Path(secure_filename(upload.filename or "garment")).stem[:80] or "garment"
                        image.save(input_dir / f"{index + 1:02d}_{stem}.png")
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                    raise ValueError("One of the uploads is not a readable image.") from None
            images = pipeline(run_dir, options)
            for item in images:
                item["artifacts"] = {
                    key: f"/api/runs/{run_id}/{path}"
                    for key, path in item["artifacts"].items()
                }
            return jsonify(run_id=run_id, images=images,
                           measurements_csv=f"/api/runs/{run_id}/output/measurements.csv",
                           metadata=f"/api/runs/{run_id}/output/metadata.csv")
        except ValueError as error:
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify(error=str(error)), 400
        except Exception:
            app.logger.exception("Measurement failed for run %s", run_id)
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify(error="Measurement failed. Check the server terminal for details."), 500
        finally:
            pipeline_lock.release()

    @app.get("/api/runs/<run_id>/<path:filename>")
    def artifact(run_id, filename):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            return jsonify(error="Run not found."), 404
        return send_from_directory(runs_dir / run_id, filename)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error="The upload exceeds 32 MB. Choose fewer or smaller images."), 413

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    return app


def main():
    parser = argparse.ArgumentParser(description="GarmentIQ local web demo")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--reload", action="store_true", help="Reload Python after source edits")
    args = parser.parse_args()
    app = create_app(args.data_dir, args.model_dir, args.device)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=args.reload, threaded=True)


if __name__ == "__main__":
    main()
