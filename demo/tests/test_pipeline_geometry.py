import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from garmentiq.landmark import derive
from garmentiq.landmark.derivation.derivation_dict import derivation_dict
from garmentiq.segmentation.change_background_color import change_background_color
from garmentiq.tailor import tailor


class PipelineGeometryTests(unittest.TestCase):
    def test_exact_intersection_bypasses_mask_and_legacy_mode_keeps_nearest_edge(self):
        coords = np.array([[[5, 1], [0, 0], [0, 5], [0, 8], [10, 8]]], dtype=float)
        rule = {"function": "derive_keypoint_coord", "direction": "parallel",
                "p1_id": "1", "p2_id": "2", "p3_id": "3", "p4_id": "4", "p5_id": "5"}
        data = {"test": {"landmarks": {"6": {"predefined": False, "derivation": rule}}}}
        mask = np.zeros((15, 15), dtype=np.uint8)
        mask[2:11, 2:11] = 255
        for mode in (None, "nearest_edge"):
            if mode: rule["intersection_mode"] = mode
            points, _ = derive("test", data, derivation_dict, landmark_coords=coords, np_mask=mask)
            self.assertEqual(points["6"], (5.0, 10.0))
        rule["intersection_mode"] = "exact"
        with patch("garmentiq.landmark.derivation.derive_keypoint_coord._get_mask_boundary") as boundary:
            points, _ = derive("test", data, derivation_dict, landmark_coords=coords)
            self.assertEqual(points["6"], (5.0, 8.0))
            boundary.assert_not_called()

    def test_derived_point_follows_mask_boundary_instead_of_template_coordinate(self):
        coords = np.array([[[5, 1], [0, 0], [0, 5], [0, 8], [10, 8]]], dtype=float)
        data = {"test": {"landmarks": {"6": {"predefined": False, "x": 999, "y": 999, "derivation": {
            "function": "derive_keypoint_coord", "direction": "parallel",
            "p1_id": "1", "p2_id": "2", "p3_id": "3", "p4_id": "4", "p5_id": "5",
        }}}}}
        for bottom in (10, 12):
            mask = np.zeros((15, 15), dtype=np.uint8)
            mask[2:bottom+1, 2:11] = 255
            points, result = derive("test", data, derivation_dict, landmark_coords=coords, np_mask=mask)
            self.assertEqual(points["6"], (5.0, float(bottom)))
            self.assertEqual(result["test"]["landmarks"]["6"]["y"], bottom)
        self.assertEqual(data["test"]["landmarks"]["6"]["y"], 999)
        coords[0, 3:] = [[2, 0], [2, 8]]
        with self.assertRaisesRegex(ValueError, "reference lines"):
            derive("test", data, derivation_dict, landmark_coords=coords, np_mask=mask)

    def test_background_replacement_thresholds_soft_masks(self):
        original = np.full((1, 6, 3), 200, dtype=np.uint8)
        for mask in (np.array([[0, 2, 64, 127, 128, 255]], dtype=np.uint8),
                     np.array([[0, .01, .25, .5, .51, 1]], dtype=float)):
            result = change_background_color(original, mask, (12, 34, 56))
            np.testing.assert_array_equal(result[0, :4], np.tile([12, 34, 56], (4, 1)))
            np.testing.assert_array_equal(result[0, 4:], original[0, 4:])
        np.testing.assert_array_equal(original, 200)

    def test_detection_and_annotated_output_use_selected_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = np.full((100, 100, 3), 200, dtype=np.uint8)
            Image.fromarray(original).save(root / "shirt.png")
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[30:80, 30:80] = 255
            background = (12, 34, 56)
            modified = change_background_color(original, mask, background)
            detection = {"top": {"landmarks": {
                "1": {"predefined": True, "description": "start", "x": 40, "y": 40, "conf": .9},
                "2": {"predefined": True, "description": "end", "x": 60, "y": 40, "conf": .8},
            }, "measurements": {"width": {"landmarks": {"start": "1", "end": "2"}, "description": "custom"}}}}
            before = copy.deepcopy(detection)
            seen = []
            agent = tailor.__new__(tailor)
            agent.input_dir = str(root)
            agent.output_dir = str(root / "output")
            agent.do_derive = agent.do_refine = agent.do_matte = False
            agent.segmentation_model_args = {"background_color": background}
            agent.classify = lambda **kw: ("top", [1])
            agent.segment = lambda **kw: (original, mask, modified)
            def detect(**kwargs):
                seen.append(kwargs["image"].copy())
                return None, None, detection
            agent.detect = detect
            metadata, _ = agent.measure(save_segmentation_image=True, save_measurement_image=True)
            np.testing.assert_array_equal(seen[0][0, 0], background)
            with Image.open(metadata.iloc[0]["measurement_image"]) as image:
                self.assertEqual(image.getpixel((0, 0)), background)
            record = json.loads(Path(metadata.iloc[0]["measurement_json"]).read_text())["shirt.png"]
            self.assertEqual(record["landmarks"]["1"]["description"], "start")
            self.assertEqual(record["measurements"]["width"]["distance"], 20)
            self.assertEqual(detection, before)
