#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import os
import pprint
import json

from PIL import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import _init_paths
from config import cfg, update_config
from core.inference import get_final_preds
from utils.transforms import transform_preds  # not used if get_final_preds already applies it
from utils.utils import create_logger
import dataset
import models
from models.tiny_vit import tinyViT

# maps classifier index (0–8) → detector category_id (1–13)
CLS_TO_DET = {
    0: 11,  # long sleeve dress
    1:  2,  # long sleeve top
    2: 10,  # short sleeve dress
    3:  1,  # short sleeve top
    4:  7,  # shorts
    5:  9,  # skirt
    6:  8,  # trousers
    7:  5,  # vest
    8: 12,  # vest dress
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run inference on Zara dataset with auto‑classification'
    )
    parser.add_argument('--cfg',           required=True, type=str)
    parser.add_argument('--model_file',    required=True, type=str,
                        help='path to your HRNet .pth')
    parser.add_argument('--cls_model_file', required=True, type=str,
                        help='path to your tiny_vit.pt')
    parser.add_argument('--data_dir',      default='',    type=str)
    parser.add_argument('--modelDir',      default='',    type=str)
    parser.add_argument('--logDir',        default='',    type=str)
    parser.add_argument('--dataDir',       default='',    type=str)
    parser.add_argument('--prevModelDir',  default='',    type=str)
    parser.add_argument('--save_coords', default=False, type=bool,
                        help='save coordinates in JSON format')
    parser.add_argument('opts',           nargs=argparse.REMAINDER)
    
    return parser.parse_args()


def main():
    args = parse_args()
    update_config(cfg, args)

    # allow overriding DATASET.ROOT from CLI
    cfg.defrost()
    if args.data_dir:
        cfg.DATASET.ROOT = args.data_dir
    cfg.freeze()

    logger, final_output_dir, _ = create_logger(cfg, args.cfg, 'valid')
    logger.info("Args:\n" + pprint.pformat(vars(args)))
    logger.info("Config:\n" + pprint.pformat(cfg))

    cudnn.benchmark     = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled       = cfg.CUDNN.ENABLED

    # ───────────
    # 1) Load HRNet for landmarks
    # ───────────
    pose_model = eval(f"models.{cfg.MODEL.NAME}.get_pose_net")(cfg, is_train=False)
    pose_model.load_state_dict(torch.load(args.model_file), strict=False)
    pose_model = torch.nn.DataParallel(pose_model).cuda()
    pose_model.eval()

    # ───────────
    # 2) Load tinyViT classifier
    # ───────────
    cls_model = tinyViT(
        num_classes = cfg.CLASSIFIER.NUM_CLASSES,
        img_size    = tuple(cfg.CLASSIFIER.IMG_SIZE),
        patch_size  = cfg.CLASSIFIER.PATCH_SIZE
    )
    state = torch.load(args.cls_model_file, map_location='cpu')
    # strip any "module." prefix
    from collections import OrderedDict
    new_state = OrderedDict({
        k.replace("module.", ""): v for k, v in state.items()
    })
    cls_model.load_state_dict(new_state, strict=False)
    cls_model = torch.nn.DataParallel(cls_model).cuda()
    cls_model.eval()

    # ───────────
    # 3) Transforms
    # ───────────
    hrnet_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    cls_tf = transforms.Compose([
        transforms.Resize(tuple(cfg.CLASSIFIER.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])

    # ───────────
    # 4) Build dataset once
    # ───────────
    valid_dataset = eval(f"dataset.{cfg.DATASET.DATASET}")(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, hrnet_tf
    )

    # ───────────
    # 5) Loop images
    # ───────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, entry in enumerate(valid_dataset.db):
        img_path = entry['image']
        print(f"\n[{idx}] Processing {img_path}")

        # 5.1) classify category
        pil = Image.open(img_path).convert("RGB")
        inp = cls_tf(pil).unsqueeze(0).cuda()
        with torch.no_grad():
            logits = cls_model(inp)
        cls_idx      = logits.argmax(1).item()    # 0…8
        category_id  = CLS_TO_DET[cls_idx]        # 1…13
        print(f" → Predicted class idx {cls_idx} → category_id {category_id}")

        # 5.2) detect landmarks
        hr_input, _, _, meta = valid_dataset[idx]
        hr_input = hr_input.unsqueeze(0).cuda()
        c = meta['center']    # already numpy array
        s = meta['scale']

        with torch.no_grad():
            output = pose_model(hr_input)

        # mask out other classes
        mask = torch.zeros_like(output).float().cuda()
        rg   = valid_dataset.gt_class_keypoints_dict[category_id]
        idxs = torch.arange(rg[0], rg[1], device=mask.device)
        mask[:, idxs, :, :] = 1
        output = output[:, rg[0]:rg[1], :, :] 

        # get_final_preds returns (x,y) already mapped into image coords
        preds_local, maxvals = get_final_preds(
            cfg, output.detach().cpu().numpy(), c, s
        )

        # Transform back from heatmap coordinate to image coordinate
        preds = preds_local.copy()
        for i in range(preds_local.shape[0]):
            preds[i] = transform_preds(
                preds_local[i], c, s, 
                [cfg.MODEL.HEATMAP_SIZE[0], cfg.MODEL.HEATMAP_SIZE[1]]
            )

        # 5.3) draw & collect valid coords
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        valid_coords = []
        for j, (x, y) in enumerate(preds[0]):
            conf = float(maxvals[0][j][0])
            if conf > 0.5:
                cv2.circle(rgb, (int(x), int(y)), 4, (0,255,0), -1)
                cv2.putText(rgb, str(j+1), (int(x)+5, int(y)-5),
                            font, 0.5, (0,255,0), 1, cv2.LINE_AA)
                valid_coords.append({
                    'keypoint_id': j+1,
                    'x': float(x),
                    'y': float(y),
                    'confidence': conf
                })

        # 5.4) Save into class folders
        cls_folder      = f"class_{category_id:02d}"
        img_out_dir     = os.path.join(final_output_dir, cls_folder, "images")
        coords_out_dir  = os.path.join(final_output_dir, cls_folder, "coordinates")
        os.makedirs(img_out_dir,    exist_ok=True)
        os.makedirs(coords_out_dir, exist_ok=True)

        # save the overlaid image
        out_img_path = os.path.join(img_out_dir, f"pred_{os.path.basename(img_path)}")
        cv2.imwrite(out_img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"   ↳ saved image → {out_img_path}")

        # save coordinates JSON
        if args.save_coords and valid_coords:
            coord_fname   = f"coords_{os.path.basename(img_path)}.json"
            out_coord_path = os.path.join(coords_out_dir, coord_fname)
            with open(out_coord_path, 'w') as f:
                json.dump({
                    'image_path': img_path,
                    'category_id': category_id,
                    'coordinates': valid_coords
                }, f, indent=2)
                
            print(f"   ↳ saved coords → {out_coord_path}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
