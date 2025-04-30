import os
import cv2
import numpy as np
from dataset.JointsDataset import JointsDataset  

class ZaraDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None,selected_class=2):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.selected_class = selected_class

        # Set the image directory that contains subfolders for each category
        self.image_dir = os.path.join(root, image_set, 'image')
        # Instead of a flat list, we'll now get the subfolder names
        all_subfolders = sorted([
            d for d in os.listdir(self.image_dir)
            if os.path.isdir(os.path.join(self.image_dir, d))
        ])
        if self.selected_class is not None:
            target_folder = f"class_{self.selected_class:02d}"
            self.subfolders = [d for d in all_subfolders if d == target_folder]
        else:
            self.subfolders = all_subfolders
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.aspect_ratio = self.image_size[0] / self.image_size[1]
        self.pixel_std = 200

        # Build the database by iterating over subfolders
        self.db = self._get_db()
        # Dummy class-to-joints mapping (ensure these ranges match your keypoints)
        self.num_joints = 294
        self.gt_class_keypoints_dict = {
            1: (0, 25), 2: (25, 58), 3: (58, 89),
            4: (89, 128), 5: (128, 143), 6: (143, 158),
            7: (158, 168), 8: (168, 182), 9: (182, 190),
            10: (190, 219), 11: (219, 256), 12: (256, 275),
            13: (275, 294)
        }
        # Optional: add dummy flip_pairs if your evaluation code expects it
        self.flip_pairs = [[] for _ in range(13)]

    def _get_db(self):
        db = []
        # Iterate over each subfolder in self.image_dir
        for subfolder in self.subfolders:
            # Only process folders that follow the naming convention "class_XX"
            if not subfolder.startswith("class_"):
                continue
            # Extract the category string (e.g., "01") and convert to int
            category_str = subfolder[len("class_"):]
            try:
                category_id = int(category_str)
            except ValueError:
                # Skip folders with unexpected naming
                continue

            # Full path to the subfolder containing images for this category
            folder_path = os.path.join(self.image_dir, subfolder)
            image_list = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.png'))
            ])

            # Process each image in this subfolder
            for i, img_name in enumerate(image_list):
                image_path = os.path.join(folder_path, img_name)
                img = cv2.imread(image_path)
                if img is None:
                    continue
                height, width, _ = img.shape
                # Use the whole image as the dummy bounding box
                bbox = [0, 0, width, height]

                center, scale = self._box2cs(bbox)
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)

                db.append({
                    'image': image_path,
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'bbox': bbox,
                    'image_id': f"{subfolder}_{i}",
                    'score': 1.0,
                    'category_id': category_id
                })
        return db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w / self.pixel_std, h / self.pixel_std], dtype=np.float32)
        scale = scale * 1.25

        return center, scale

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        print("No evaluation metric for ZaraDataset — skipping evaluation.")
        return {}, 0.0  # empty dict and dummy perf indicator

