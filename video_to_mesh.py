import argparse
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import garmentiq as giq
from garmentiq.classification.model_definition import tinyViT
from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
from garmentiq.garment_classes import garment_classes


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception as e:
        raise RuntimeError(
            "open3d is required for mesh export.\n"
            "Install it with: pip install open3d\n"
            f"Original import error: {e}"
        )


def _require_transformers_pipeline():
    try:
        from transformers import pipeline  # type: ignore

        return pipeline
    except Exception as e:
        raise RuntimeError(
            "transformers is required for depth estimation.\n"
            "Install it with: pip install transformers\n"
            f"Original import error: {e}"
        )


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


def estimate_depth(depth_pipe, frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = depth_pipe(frame_rgb)
    depth = out["depth"]
    depth_np = np.array(depth).astype(np.float32)
    return depth_np


def estimate_mask(segmenter, temp_image_path: str) -> np.ndarray:
    _, mask = giq.segmentation.extract(
        model=segmenter,
        image_path=temp_image_path,
        resize_dim=(1024, 1024),
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        high_precision=True,
    )
    return mask


def detect_keypoints(landmark_model, class_name: str, frame_rgb: np.ndarray):
    coords, _, detection_dict = giq.landmark.detect(
        class_name=class_name,
        class_dict=garment_classes,
        image_path=frame_rgb,
        model=landmark_model,
        scale_std=200.0,
        resize_dim=[288, 384],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    )
    return coords, detection_dict


def classify_frame(classes, classifier, temp_image_path: str) -> str:
    label, _ = giq.classification.predict(
        model=classifier,
        image_path=temp_image_path,
        classes=classes,
        resize_dim=(120, 184),
        normalize_mean=[0.8047, 0.7808, 0.7769],
        normalize_std=[0.2957, 0.3077, 0.3081],
        verbose=False,
    )
    return label


def compute_default_intrinsics(width: int, height: int) -> CameraIntrinsics:
    f = float(max(width, height))
    return CameraIntrinsics(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)


def masked_depth_to_points(
    depth: np.ndarray,
    mask: np.ndarray,
    intr: CameraIntrinsics,
    depth_scale: float,
    stride: int,
) -> np.ndarray:
    h, w = depth.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    binary = mask > 127
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    step = max(1, stride)
    xs = xs[::step]
    ys = ys[::step]
    z = depth[ys, xs] * float(depth_scale)

    valid = np.isfinite(z) & (z > 0)
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = (xs.astype(np.float32) - intr.cx) * z / intr.fx
    y = (ys.astype(np.float32) - intr.cy) * z / intr.fy
    pts = np.stack([x, y, z.astype(np.float32)], axis=1)
    return pts


def fuse_pointcloud(points_list: list[np.ndarray]) -> np.ndarray:
    if not points_list:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.concatenate([p for p in points_list if p.size > 0], axis=0)
    if pts.size == 0:
        return pts
    return pts


def save_mesh_from_points(
    points_xyz: np.ndarray,
    out_path: str,
    voxel_size: float,
    poisson_depth: int,
):
    o3d = _require_open3d()
    if points_xyz.shape[0] < 1000:
        raise RuntimeError(
            f"Not enough 3D points to reconstruct a mesh ({points_xyz.shape[0]} points). "
            "Try increasing --frames or decreasing --stride."
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4.0, max_nn=30)
        if voxel_size > 0
        else o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(50)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(poisson_depth)
    )

    densities_np = np.asarray(densities)
    if densities_np.size:
        thresh = np.quantile(densities_np, 0.02)
        vertices_to_remove = densities_np < thresh
        mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok = o3d.io.write_triangle_mesh(out_path, mesh)
    if not ok:
        raise RuntimeError(f"Failed to write mesh to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="GarmentIQ: video -> keypoints + depth -> fused 3D mesh export (MVP)"
    )
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument(
        "--model-dir",
        default=os.path.join(os.getcwd(), "models"),
        help="Directory containing tiny_vit_inditex_finetuned.pt and hrnet.pth",
    )
    p.add_argument(
        "--garment-class",
        default="",
        help="Optional: skip per-frame classification by providing class name (e.g. 'skirt')",
    )
    p.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face depth-estimation model id",
    )
    p.add_argument("--frames", type=int, default=60, help="Max frames to sample from video")
    p.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Point sampling stride inside garment mask (higher=faster, fewer points)",
    )
    p.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="Scale factor applied to depth map values to get ~meters (varies by depth model)",
    )
    p.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling before meshing (meters-ish)",
    )
    p.add_argument(
        "--poisson-depth",
        type=int,
        default=9,
        help="Poisson reconstruction depth (higher=more detail, slower)",
    )
    p.add_argument("--out", default="output/mesh/video_mesh.ply", help="Output mesh path (.ply/.obj)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    for fname in ["tiny_vit_inditex_finetuned.pt", "hrnet.pth"]:
        fpath = os.path.join(args.model_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required model file: {fpath}")

    pipeline = _require_transformers_pipeline()
    depth_pipe = pipeline("depth-estimation", model=args.depth_model)

    classes, classifier, segmenter, landmark_model = build_models(args.model_dir)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read the first frame from video")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    intr = compute_default_intrinsics(width, height)

    sample_count = max(1, int(args.frames))
    if total_frames > 0:
        idxs = np.linspace(0, total_frames - 1, num=sample_count).astype(int).tolist()
    else:
        idxs = list(range(sample_count))

    points_accum: list[np.ndarray] = []
    last_keypoints: Optional[Tuple[np.ndarray, dict]] = None

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_image_path = os.path.join(tmpdir, "frame.jpg")
        for n, frame_idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            cv2.imwrite(temp_image_path, frame_bgr)

            class_name = args.garment_class.strip()
            if not class_name:
                class_name = classify_frame(classes, classifier, temp_image_path)

            mask = estimate_mask(segmenter, temp_image_path)
            depth = estimate_depth(depth_pipe, frame_bgr)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                kpts, det = detect_keypoints(landmark_model, class_name, frame_rgb)
                last_keypoints = (kpts, det)
            except Exception:
                pass

            pts = masked_depth_to_points(
                depth=depth,
                mask=mask,
                intr=intr,
                depth_scale=float(args.depth_scale),
                stride=int(args.stride),
            )
            points_accum.append(pts)

            if (n + 1) % 10 == 0:
                print(f"processed {n+1}/{len(idxs)} frames, points so far: {sum(p.shape[0] for p in points_accum)}")

    cap.release()

    fused = fuse_pointcloud(points_accum)
    print(f"total fused points: {fused.shape[0]}")
    save_mesh_from_points(
        points_xyz=fused,
        out_path=args.out,
        voxel_size=float(args.voxel_size),
        poisson_depth=int(args.poisson_depth),
    )
    print(f"mesh written to: {args.out}")

    if last_keypoints is not None:
        kpts, _ = last_keypoints
        print(f"last detected keypoints shape: {kpts.shape} (batch, num_points, 2)")


if __name__ == "__main__":
    main()

