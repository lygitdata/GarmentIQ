import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from demo.app import create_app, ROOT
from demo.calibration import calibrate_record, validate_calibrations
from demo.instructions import CATALOG, validate_instruction
from test_app import upload


class InstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "instructions"
        shutil.copytree(ROOT / "src/garmentiq/instruction", self.directory)
        self.client = create_app(data_dir=Path(self.temp.name) / "data", instruction_dir=self.directory).test_client()
        self.category = "short sleeve top"
        self.url = "/api/instructions/short%20sleeve%20top"

    def test_all_categories_match_the_model_landmark_counts(self):
        expected = [37, 33, 29, 25, 10, 8, 14, 15, 19]
        names = sorted(CATALOG)
        self.assertEqual([len(CATALOG[name]["landmarks"]) for name in names], expected)
        self.assertEqual(self.client.get("/api/instructions").get_json()["categories"], names)

    def test_save_replaces_only_selected_category_and_backs_up_previous_file(self):
        original = {p.name: p.read_bytes() for p in self.directory.glob("*.json")}
        data = self.client.get(self.url).get_json()
        garment = data["instruction"][self.category]
        garment["measurements"]["custom chest"] = {"landmarks": {"start": "12", "end": "20"}, "description": "New custom width"}
        response = self.client.put(self.url, json={"instruction": data["instruction"], "revision": data["revision"]})
        self.assertEqual(response.status_code, 200, response.get_json())
        saved = response.get_json()
        self.assertIn("custom chest", json.loads((self.directory / f"{self.category}.json").read_text())[self.category]["measurements"])
        self.assertEqual(Path(saved["backup"]).read_bytes(), original[f"{self.category}.json"])
        self.assertNotEqual(saved["revision"], data["revision"])
        for name, content in original.items():
            if name != f"{self.category}.json":
                self.assertEqual((self.directory / name).read_bytes(), content)
        stale = self.client.put(self.url, json={"instruction": data["instruction"], "revision": data["revision"]})
        self.assertEqual(stale.status_code, 409)

    def test_invalid_derived_reference_does_not_modify_file(self):
        data = self.client.get(self.url).get_json()
        before = (self.directory / f"{self.category}.json").read_bytes()
        data["instruction"][self.category]["landmarks"]["26"]["derivation"]["p1_id"] = "26"
        response = self.client.put(self.url, json={"instruction": data["instruction"], "revision": data["revision"]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("detected landmark", response.get_json()["error"])
        self.assertEqual((self.directory / f"{self.category}.json").read_bytes(), before)

    def test_intersection_mode_is_saved_and_invalid_mode_is_rejected(self):
        data = self.client.get(self.url).get_json()
        rule = data["instruction"][self.category]["landmarks"]["26"]["derivation"]
        rule["intersection_mode"] = "exact"
        response = self.client.put(self.url, json={"instruction": data["instruction"], "revision": data["revision"]})
        self.assertEqual(response.status_code, 200, response.get_json())
        saved = self.client.get(self.url).get_json()
        self.assertEqual(saved["instruction"][self.category]["landmarks"]["26"]["derivation"]["intersection_mode"], "exact")
        other = saved["instruction"][self.category]["landmarks"]["27"]["derivation"]
        self.assertEqual(other["intersection_mode"], "nearest_edge")
        rule = saved["instruction"][self.category]["landmarks"]["26"]["derivation"]
        rule["intersection_mode"] = "invalid"
        self.assertEqual(self.client.put(self.url, json={"instruction": saved["instruction"], "revision": saved["revision"]}).status_code, 400)

    def test_rule_references_are_included_even_when_not_selected_endpoints(self):
        instruction = self.client.get(self.url).get_json()["instruction"]
        output = validate_instruction(self.category, instruction)[self.category]
        self.assertIn("4", output["landmarks"])
        self.assertIn("15", output["landmarks"])
        self.assertTrue(output["landmarks"]["4"]["predefined"])

    def test_schema_rejects_unknown_category_invalid_measurements_and_coordinates(self):
        self.assertEqual(self.client.get("/api/instructions/unknown").status_code, 404)
        original = self.client.get(self.url).get_json()["instruction"]
        for change in ("missing endpoint", "repeated endpoints", "fake predefined", "unknown function", "nonfinite"):
            data = copy.deepcopy(original)
            garment = data[self.category]
            if change == "missing endpoint": garment["measurements"]["chest"]["landmarks"]["end"] = "999"
            if change == "repeated endpoints": garment["measurements"]["chest"]["landmarks"]["end"] = "12"
            if change == "fake predefined": garment["landmarks"]["26"]["predefined"] = True
            if change == "unknown function": garment["landmarks"]["26"]["derivation"]["function"] = "eval"
            if change == "nonfinite": garment["landmarks"]["2"]["x"] = float("nan")
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_instruction(self.category, data)

    def test_calibration_form_rejects_partial_nonfinite_and_mismatched_entries(self):
        for value in ([{"width_cm": 50}], [{"width_cm": -1, "height_cm": 20}],
                      [{"width_cm": float("nan"), "height_cm": 20}], [], {}):
            response = self.client.post("/api/measure", data={"images": upload(), "calibrations": json.dumps(value)})
            self.assertEqual(response.status_code, 400, value)


class CalibrationTests(unittest.TestCase):
    def test_diagonal_uses_separate_horizontal_and_vertical_scales(self):
        record = {"landmarks": {"1": {"x": 0, "y": 0}, "2": {"x": 30, "y": 40}},
                  "measurements": {"diagonal": {"distance": 50, "landmarks": {"start": "1", "end": "2"}}}}
        calibrate_record(record, (1000, 500), {"width_cm": 100, "height_cm": 100})
        self.assertAlmostEqual(record["measurements"]["diagonal"]["distance_cm"], math.sqrt(73))
        self.assertEqual(record["measurements"]["diagonal"]["distance_px"], 50)
        self.assertEqual(record["landmarks"]["2"]["x_cm"], 3)
        self.assertEqual(record["landmarks"]["2"]["y_cm"], 8)

    def test_unscaled_records_keep_pixel_units(self):
        record = {"landmarks": {}, "measurements": {"length": {"distance": 5}}}
        calibrate_record(record, (10, 20), None)
        self.assertIsNone(record["calibration"])
        self.assertNotIn("distance_cm", record["measurements"]["length"])
        self.assertEqual(validate_calibrations([None, {"width_cm": 10, "height_cm": 20}], 2),
                         [None, {"width_cm": 10.0, "height_cm": 20.0}])
