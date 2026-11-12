"""API checks without downloading or loading neural networks."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from demo.app import create_app
from demo.pipeline import instructions_for_run


def upload(name="shirt.jpg"):
    data = BytesIO()
    Image.new("RGB", (24, 32), "white").save(data, "JPEG")
    data.seek(0)
    return data, name


class LocalDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []

        def runner(run_dir, options):
            self.calls.append((run_dir, options))
            (run_dir / "output").mkdir()
            (run_dir / "output" / "metadata.csv").write_text("filename,class\n")
            image = next((run_dir / "input").iterdir())
            return [{"filename": image.name, "class": "short sleeve top",
                     "measurements": {"chest": {"distance": 12.5}},
                     "artifacts": {"original": f"input/{image.name}"}}]

        self.app = create_app(data_dir=self.temp.name, runner=runner)
        self.app.testing = True
        self.client = self.app.test_client()

    def test_page_and_setup_status_without_loading_models(self):
        with self.client.get("/") as response:
            self.assertEqual(response.status_code, 200)
        status = self.client.get("/api/health").get_json()
        self.assertFalse(status["ready"])
        self.assertIn("hrnet.pth", status["missing_models"])

    def test_measure_and_download_with_actual_colour_and_options(self):
        response = self.client.post("/api/measure", data={
            "images": upload(), "background": "#1234ab", "derive": "false", "refine": "true",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.calls[0][1], {"background": [18, 52, 171], "derive": False, "refine": True})
        result = response.get_json()
        with self.client.get(result["images"][0]["artifacts"]["original"]) as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "image/png")
        with self.client.get(result["metadata"]) as response:
            self.assertEqual(response.status_code, 200)

    def test_each_run_is_isolated_and_filenames_are_sanitized(self):
        results = [self.client.post("/api/measure", data={"images": upload("../../shirt.jpg")}).get_json() for _ in range(2)]
        self.assertNotEqual(results[0]["run_id"], results[1]["run_id"])
        for result in results:
            self.assertEqual(result["images"][0]["filename"], "01_shirt.png")
        self.assertEqual(len(list((Path(self.temp.name) / "runs").iterdir())), 2)

    def test_missing_invalid_and_excess_uploads(self):
        self.assertEqual(self.client.post("/api/measure").status_code, 400)
        self.assertEqual(self.client.post("/api/measure", data={"images": (BytesIO(b"not an image"), "bad.png")}).status_code, 400)
        self.assertEqual(self.client.post("/api/measure", data={"images": [upload() for _ in range(13)]}).status_code, 400)
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(list((Path(self.temp.name) / "runs").iterdir()), [])

    def test_invalid_options_are_rejected(self):
        for options in [{"background": "red"}, {"derive": "invalid"}, {"refine": "invalid"}]:
            with self.subTest(options=options):
                response = self.client.post("/api/measure", data={"images": upload(), **options})
                self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.calls), 0)

    def test_size_limit_is_reported_as_json(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 100
        response = self.client.post("/api/measure", data={"images": upload()})
        self.assertEqual(response.status_code, 413)
        self.assertIn("32 MB", response.get_json()["error"])

    def test_files_cannot_escape_the_run_directory(self):
        response = self.client.post("/api/measure", data={"images": upload()}).get_json()
        for path in ["../../test_app.py", "../outside.txt"]:
            self.assertEqual(self.client.get(f"/api/runs/{response['run_id']}/{path}").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/invalid/input/file.png").status_code, 404)

    def test_failure_releases_pipeline_for_retry(self):
        # A failed input must not hold the inference lock forever.
        self.client.post("/api/measure", data={"images": (BytesIO(b"broken"), "bad.jpg")})
        self.assertEqual(self.client.post("/api/measure", data={"images": upload()}).status_code, 200)

    def test_overlapping_measurements_are_rejected(self):
        entered, release = threading.Event(), threading.Event()

        def blocking_runner(run_dir, options):
            entered.set()
            release.wait(timeout=5)
            return []

        app = create_app(data_dir=self.temp.name, runner=blocking_runner)
        responses = []
        thread = threading.Thread(target=lambda: responses.append(
            app.test_client().post("/api/measure", data={"images": upload()})
        ))
        thread.start()
        try:
            self.assertTrue(entered.wait(timeout=5))
            response = app.test_client().post("/api/measure", data={"images": upload()})
            self.assertEqual(response.status_code, 409)
        finally:
            release.set()
            thread.join(timeout=5)
        self.assertEqual(responses[0].status_code, 200)

    def test_missing_models_return_actionable_setup_error(self):
        app = create_app(data_dir=self.temp.name)
        with patch("demo.app.runtime_status", return_value={"ready": False, "missing_packages": [], "missing_models": ["hrnet.pth"]}):
            response = app.test_client().post("/api/measure", data={"images": upload()})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["missing_models"], ["hrnet.pth"])

    def test_disable_derivation_removes_template_measurements(self):
        path = Path(self.temp.name) / "top.json"
        schema = {"top": {"landmarks": {
            "1": {"predefined": True, "x": 0, "y": 0},
            "2": {"predefined": True, "x": 3, "y": 4},
            "3": {"predefined": False, "x": 999, "y": 999},
        }, "measurements": {
            "detected": {"landmarks": {"start": "1", "end": "2"}},
            "derived": {"landmarks": {"start": "1", "end": "3"}},
        }}}
        path.write_text(json.dumps(schema))
        classes = {"top": {"instruction": str(path)}}
        filtered = instructions_for_run(classes, self.temp.name, derive=False)
        output = json.loads(Path(filtered["top"]["instruction"]).read_text())
        self.assertEqual(set(output["top"]["measurements"]), {"detected"})
        self.assertEqual(set(output["top"]["landmarks"]), {"1", "2"})
        self.assertEqual(json.loads(path.read_text()), schema)
        self.assertEqual(classes["top"]["instruction"], str(path))


if __name__ == "__main__":
    unittest.main()
