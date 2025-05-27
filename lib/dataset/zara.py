import os
import cv2
import numpy as np
from dataset.JointsDataset import JointsDataset


class ZaraDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.image_dir = os.path.join(root, image_set, 'image')
        self.image_list = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.aspect_ratio = self.image_size[0] / self.image_size[1]
        self.pixel_std = 200

        self.db = self._get_db()
        # Dummy class-to-joints mapping
        self.num_joints = 294
        self.gt_class_keypoints_dict = {1: (0, 25), 2: (25, 58), 3: (58, 89),
                4: (89, 128), 5: (128, 143), 6: (143, 158), 7: (158, 168),
                8: (168, 182), 9: (182, 190), 10: (190, 219),
                11: (219, 256), 12: (256, 275), 13: (275, 294)}

        
    def _get_db(self):
        db = []
        for i, img_name in enumerate(self.image_list):
            image_path = os.path.join(self.image_dir, img_name)

            # Dummy bounding box: whole image
            img = cv2.imread(image_path)
            height, width, _ = img.shape
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
                'image_id': i,
                'score': 1.0,
                'category_id': 2
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
