import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import garmentiq as giq
from garmentiq.classification.model_definition import tinyViT
from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
from garmentiq.garment_classes import garment_classes


@dataclass
class InferenceResult:
    label: str
    frame: np.ndarray
    mask: np.ndarray
    coords: np.ndarray
    detection_dict: dict


def build_models(model_dir: str):
    classes = sorted(list(garment_classes.keys()))
    classifier = giq.classification.load_model(
        model_path=os.path.join(model_dir, "tiny_vit_inditex_finetuned.pt"),
        model_class=tinyViT,
        model_args={
            "num_classes": len(classes),
            "img_size": (120, 184),
            "patch_size": 6,
        },
    )
    segmenter = giq.segmentation.load_model(
        pretrained_model="lygitdata/BiRefNet_garmentiq_backup",
        pretrained_model_args={"trust_remote_code": True},
        high_precision=True,
    )
    landmark_model = giq.landmark.detection.load_model(
        model_path=os.path.join(model_dir, "hrnet.pth"),
        model_class=PoseHighResolutionNet(),
    )
    return classes, classifier, segmenter, landmark_model


def run_inference_on_frame(
    frame_bgr: np.ndarray,
    classes: list[str],
    classifier,
    segmenter,
    landmark_model,
    temp_image_path: str,
) -> InferenceResult:
    cv2.imwrite(temp_image_path, frame_bgr)

    label, _ = giq.classification.predict(
        model=classifier,
        image_path=temp_image_path,
        classes=classes,
        resize_dim=(120, 184),
        normalize_mean=[0.8047, 0.7808, 0.7769],
        normalize_std=[0.2957, 0.3077, 0.3081],
        verbose=False,
    )

    original_img, mask = giq.segmentation.extract(
        model=segmenter,
        image_path=temp_image_path,
        resize_dim=(1024, 1024),
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        high_precision=True,
    )

    coords, _, detection_dict = giq.landmark.detect(
        class_name=label,
        class_dict=garment_classes,
        image_path=temp_image_path,
        model=landmark_model,
        scale_std=200.0,
        resize_dim=[288, 384],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    )

    return InferenceResult(
        label=label,
        frame=original_img,
        mask=mask,
        coords=coords,
        detection_dict=detection_dict,
    )


def draw_landmarks_overlay(frame_rgb: np.ndarray, coords: np.ndarray, label: str, fps: float):
    vis = frame_rgb.copy()
    for pt in coords[0]:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

    cv2.putText(
        vis,
        f"class: {label}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"display fps: {fps:.2f}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def project_points(points_3d: np.ndarray, angle_deg: float, canvas_size: int = 700) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    rot_y = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=np.float32,
    )
    rotated = points_3d @ rot_y.T
    depth = rotated[:, 2] + 2.0
    projected = rotated[:, :2] / depth[:, None]
    projected = (projected * (canvas_size * 0.9)) + (canvas_size / 2.0)
    return projected.astype(np.int32)


def render_3d_canvas(mask: np.ndarray, coords: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = mask.shape[:2]
    canvas = np.zeros((700, 700, 3), dtype=np.uint8)
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        cv2.putText(
            canvas,
            "No garment mask detected",
            (170, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return canvas

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    max_dist = float(dist.max()) if float(dist.max()) > 0 else 1.0

    ys, xs = np.where(binary > 0)
    stride = max(1, len(xs) // 3500)
    xs_s = xs[::stride]
    ys_s = ys[::stride]
    z_s = dist[ys_s, xs_s] / max_dist

    x_n = (xs_s.astype(np.float32) - (w / 2.0)) / max(w, 1)
    y_n = -((ys_s.astype(np.float32) - (h / 2.0)) / max(h, 1))
    pts3d = np.stack([x_n, y_n, z_s.astype(np.float32)], axis=1)

    pts2d = project_points(pts3d, angle_deg=angle_deg, canvas_size=700)
    valid = (
        (pts2d[:, 0] >= 0)
        & (pts2d[:, 0] < 700)
        & (pts2d[:, 1] >= 0)
        & (pts2d[:, 1] < 700)
    )
    canvas[pts2d[valid, 1], pts2d[valid, 0]] = (120, 180, 255)

    lm = coords[0]
    lm_x = np.clip(lm[:, 0].astype(np.int32), 0, w - 1)
    lm_y = np.clip(lm[:, 1].astype(np.int32), 0, h - 1)
    lm_z = dist[lm_y, lm_x] / max_dist
    lm_xn = (lm_x.astype(np.float32) - (w / 2.0)) / max(w, 1)
    lm_yn = -((lm_y.astype(np.float32) - (h / 2.0)) / max(h, 1))
    lm_3d = np.stack([lm_xn, lm_yn, lm_z.astype(np.float32)], axis=1)
    lm_2d = project_points(lm_3d, angle_deg=angle_deg, canvas_size=700)
    for p in lm_2d:
        if 0 <= p[0] < 700 and 0 <= p[1] < 700:
            cv2.circle(canvas, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)

    cv2.putText(
        canvas,
        "Pseudo-3D garment canvas",
        (190, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Keys: q=quit, a/d=rotate, r=reset",
        (145, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


def parse_source(src: str):
    if src.isdigit():
        return int(src)
    return src


def main():
    parser = argparse.ArgumentParser(
        description="GarmentIQ real-time video to pseudo-3D canvas MVP"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index (e.g. 0) or file path",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(os.getcwd(), "models"),
        help="Directory containing tiny_vit_inditex_finetuned.pt and hrnet.pth",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=10,
        help="Run heavy inference every N frames (default: 10)",
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.model_dir, "tiny_vit_inditex_finetuned.pt")):
        raise FileNotFoundError(
            f"Missing classification model: {os.path.join(args.model_dir, 'tiny_vit_inditex_finetuned.pt')}"
        )
    if not os.path.exists(os.path.join(args.model_dir, "hrnet.pth")):
        raise FileNotFoundError(
            f"Missing landmark model: {os.path.join(args.model_dir, 'hrnet.pth')}"
        )

    classes, classifier, segmenter, landmark_model = build_models(args.model_dir)
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    last: Optional[InferenceResult] = None
    frame_idx = 0
    angle = 25.0
    prev_t = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_image_path = os.path.join(tmpdir, "frame.jpg")
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_idx += 1
            if frame_idx % max(args.process_every, 1) == 0:
                try:
                    last = run_inference_on_frame(
                        frame_bgr=frame_bgr,
                        classes=classes,
                        classifier=classifier,
                        segmenter=segmenter,
                        landmark_model=landmark_model,
                        temp_image_path=temp_image_path,
                    )
                except Exception as e:
                    print(f"[WARN] inference failed on frame {frame_idx}: {e}")

            now_t = time.time()
            fps = 1.0 / max(now_t - prev_t, 1e-6)
            prev_t = now_t

            if last is not None:
                overlay = draw_landmarks_overlay(last.frame, last.coords, last.label, fps)
                canvas3d = render_3d_canvas(last.mask, last.coords, angle)
            else:
                overlay = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                cv2.putText(
                    overlay,
                    "Waiting for first inference...",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                canvas3d = np.zeros((700, 700, 3), dtype=np.uint8)

            left = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            left = cv2.resize(left, (700, 700))
            combined = np.hstack([left, canvas3d])
            cv2.imshow("GarmentIQ Realtime MVP (2D -> pseudo-3D)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                angle -= 7.0
            if key == ord("d"):
                angle += 7.0
            if key == ord("r"):
                angle = 25.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
