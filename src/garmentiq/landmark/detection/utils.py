import math
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from typing import Union


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(
        batch_heatmaps, np.ndarray
    ), "batch_heatmaps should be numpy.ndarray"
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(output, height=96, width=72):
    heatmap_height = height
    heatmap_width = width

    batch_heatmaps = output
    coords, maxvals = get_max_preds(batch_heatmaps)
    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array(
                    [hm[py][px + 1] - hm[py][px - 1], hm[py + 1][px] - hm[py - 1][px]]
                )
                coords[n][p] += np.sign(diff) * 0.25

    return coords, maxvals


def flip_back(output_flipped, matched_parts, heatmap_wid):
    if output_flipped.ndim == 4:
        output_flipped = output_flipped[:, :, :, ::-1]
        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp
    elif output_flipped.ndim == 3:
        output_flipped[:, :, 0] = heatmap_wid - output_flipped[:, :, 0]
        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :].copy()
            output_flipped[:, pair[0], :] = output_flipped[:, pair[1], :]
            output_flipped[:, pair[1], :] = tmp
    else:
        raise NotImplementedError(
            "output_flipped should be [batch_size, num_joints, height, width], "
            "or [batch_size, num_joints, coord_dim"
        )

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = (
            joints[pair[1], :],
            joints[pair[0], :].copy(),
        )
        joints_vis[pair[0], :], joints_vis[pair[1], :] = (
            joints_vis[pair[1], :],
            joints_vis[pair[0], :].copy(),
        )

    return joints * joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size: list[int, int] = [72, 96]):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])), flags=cv2.INTER_LINEAR
    )

    return dst_img


def input_image_transform(
    img_input: Union[str, np.ndarray],
    scale_std: float = 200.0,
    resize_dim: list[int] = [288, 384],
    normalize_mean: list[float] = [0.485, 0.456, 0.406],
    normalize_std: list[float] = [0.229, 0.224, 0.225],
):
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input.astype(np.uint8))
    else:
        raise ValueError("img_input must be a file path or a NumPy array.")

    image_np = np.array(img)
    
    h, w = image_np.shape[:2]
    center = np.array([w / 2, h / 2], dtype=np.float32)
    scale = np.array([w / scale_std, h / scale_std], dtype=np.float32)
    image_size = np.array(resize_dim)
    rotation = 0

    trans = get_affine_transform(
        center, scale, rotation, image_size
    )
    warped_image = cv2.warpAffine(
        image_np,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR,
    )

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(normalize_mean, normalize_std)
    input_tensor = normalize(to_tensor(warped_image)).unsqueeze(0)

    return input_tensor, image_np, center, scale
