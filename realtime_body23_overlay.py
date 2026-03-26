import argparse
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np


def _require_mediapipe():
    try:
        import mediapipe as mp  # type: ignore

        return mp
    except Exception as e:
        raise RuntimeError(
            "This script requires mediapipe.\n"
            "Install with: pip install mediapipe\n"
            f"Original import error: {e}"
        )


@dataclass(frozen=True)
class Pt:
    x: int
    y: int


def _to_px(w: int, h: int, lm) -> Pt:
    return Pt(x=int(lm.x * w), y=int(lm.y * h))


def _mid(a: Pt, b: Pt) -> Pt:
    return Pt(x=(a.x + b.x) // 2, y=(a.y + b.y) // 2)


def _draw_point(img: np.ndarray, p: Pt, label: str):
    cv2.circle(img, (p.x, p.y), 4, (0, 0, 0), -1)
    cv2.circle(img, (p.x, p.y), 3, (255, 255, 255), -1)
    cv2.putText(
        img,
        str(label),
        (p.x + 6, p.y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        str(label),
        (p.x + 6, p.y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_line(img: np.ndarray, a: Pt, b: Pt, label: str):
    cv2.line(img, (a.x, a.y), (b.x, b.y), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (a.x, a.y), (b.x, b.y), (255, 255, 255), 1, cv2.LINE_AA)
    m = _mid(a, b)
    cv2.putText(
        img,
        str(label),
        (m.x + 6, m.y + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        str(label),
        (m.x + 6, m.y + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _safe_get(points: Dict[str, Pt], key: str) -> Optional[Pt]:
    return points.get(key)


def _build_body23_points(w: int, h: int, pose_landmarks) -> Dict[str, Pt]:
    # MediaPipe Pose landmark indices (BlazePose, 33 points)
    lm = pose_landmarks.landmark
    NOSE = 0
    L_EAR, R_EAR = 7, 8
    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_WR, R_WR = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28
    L_HEEL, R_HEEL = 29, 30

    p = {
        "nose": _to_px(w, h, lm[NOSE]),
        "l_ear": _to_px(w, h, lm[L_EAR]),
        "r_ear": _to_px(w, h, lm[R_EAR]),
        "l_sh": _to_px(w, h, lm[L_SH]),
        "r_sh": _to_px(w, h, lm[R_SH]),
        "l_el": _to_px(w, h, lm[L_EL]),
        "r_el": _to_px(w, h, lm[R_EL]),
        "l_wr": _to_px(w, h, lm[L_WR]),
        "r_wr": _to_px(w, h, lm[R_WR]),
        "l_hip": _to_px(w, h, lm[L_HIP]),
        "r_hip": _to_px(w, h, lm[R_HIP]),
        "l_knee": _to_px(w, h, lm[L_KNEE]),
        "r_knee": _to_px(w, h, lm[R_KNEE]),
        "l_ank": _to_px(w, h, lm[L_ANK]),
        "r_ank": _to_px(w, h, lm[R_ANK]),
        "l_heel": _to_px(w, h, lm[L_HEEL]),
        "r_heel": _to_px(w, h, lm[R_HEEL]),
    }
    p["neck"] = _mid(p["l_sh"], p["r_sh"])
    p["waist"] = _mid(p["l_hip"], p["r_hip"])
    p["top_head"] = Pt(x=p["nose"].x, y=max(0, p["nose"].y - int(0.20 * (h / 2))))
    p["floor"] = Pt(x=p["waist"].x, y=max(p["l_heel"].y, p["r_heel"].y))
    return p


def _overlay_body23(img_bgr: np.ndarray, points: Dict[str, Pt], fps: float):
    # This is an MVP mapping to resemble the provided “23 measurements” style.
    # It is not a medically/anthropometrically calibrated measurement system.
    img = img_bgr

    # Measurement lines (approximate)
    lines: List[Tuple[str, str, str]] = [
        ("1", "top_head", "floor"),          # height
        ("23", "l_ear", "r_ear"),            # head width
        ("15", "neck", "neck"),              # neck marker (point)
        ("6", "l_sh", "r_sh"),               # shoulder width
        ("11", "l_sh", "l_wr"),              # left arm length
        ("11", "r_sh", "r_wr"),              # right arm length
        ("16", "l_hip", "l_knee"),           # thigh length
        ("16", "r_hip", "r_knee"),
        ("17", "waist", "l_heel"),           # waist-to-floor (approx)
        ("17", "waist", "r_heel"),
    ]

    for lab, a, b in lines:
        pa = _safe_get(points, a)
        pb = _safe_get(points, b)
        if pa is None or pb is None:
            continue
        if a == b:
            _draw_point(img, pa, lab)
        else:
            _draw_line(img, pa, pb, lab)

    # Key points with numeric labels (subset to resemble the sample)
    point_labels: List[Tuple[str, str]] = [
        ("23", "l_ear"),
        ("23", "r_ear"),
        ("15", "neck"),
        ("6", "l_sh"),
        ("6", "r_sh"),
        ("12", "l_el"),
        ("12", "r_el"),
        ("14", "l_wr"),
        ("14", "r_wr"),
        ("18", "l_hip"),
        ("18", "r_hip"),
        ("19", "l_knee"),
        ("19", "r_knee"),
        ("21", "l_ank"),
        ("21", "r_ank"),
    ]
    for lab, k in point_labels:
        pk = _safe_get(points, k)
        if pk is not None:
            _draw_point(img, pk, lab)

    cv2.putText(
        img,
        f"Body23 overlay (MVP) | FPS: {fps:.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Body23 overlay (MVP) | FPS: {fps:.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def parse_source(src: str):
    if src.isdigit():
        return int(src)
    return src


def main():
    ap = argparse.ArgumentParser(description="Realtime body keypoints overlay (23-style, MVP).")
    ap.add_argument("--source", default="0", help="Camera index (0) or video path")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    args = ap.parse_args()

    mp = _require_mediapipe()
    Pose = mp.solutions.pose.Pose

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    prev_t = time.time()
    with Pose(
        static_image_mode=False,
        model_complexity=int(args.model_complexity),
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)

            now_t = time.time()
            fps = 1.0 / max(now_t - prev_t, 1e-6)
            prev_t = now_t

            vis = frame_bgr.copy()
            if res.pose_landmarks is not None:
                h, w = vis.shape[:2]
                pts = _build_body23_points(w, h, res.pose_landmarks)
                vis = _overlay_body23(vis, pts, fps)
            else:
                cv2.putText(
                    vis,
                    "No pose detected",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Body23 Overlay (MVP)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

