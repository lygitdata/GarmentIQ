from __future__ import absolute_import, division, print_function

import argparse, os, pprint
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

import _init_paths
from config import cfg, update_config
from core.loss import JointsMSELoss
from core.inference import get_final_preds
from utils.transforms import transform_preds
from utils.utils import create_logger

import dataset
import models
from models.tiny_vit import tinyViT

# import classifier definition
import os
import sys



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
    parser.add_argument('opts',           nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    if args.data_dir:
        cfg.DATASET.ROOT = args.data_dir
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark     = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled       = cfg.CUDNN.ENABLED

    # ── 1) Load HRNet pose model
    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    pose_model.load_state_dict(torch.load(args.model_file), strict=False)
    pose_model = torch.nn.DataParallel(pose_model).cuda()
    pose_model.eval()

    # ── 2) Load tinyViT classifier
    #    Your tinyViT was initialized with num_classes=9, img_size=(120,184), patch_size=6
    cls_model = tinyViT(
    num_classes  = cfg.CLASSIFIER.NUM_CLASSES,
    img_size     = tuple(cfg.CLASSIFIER.IMG_SIZE),
    patch_size   = cfg.CLASSIFIER.PATCH_SIZE
)
    state = torch.load(cfg.CLASSIFIER.PRETRAINED, map_location='cpu')
    # strip "module." if necessary
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        nk = k.replace('module.', '')
        new_state[nk] = v
    cls_model.load_state_dict(new_state, strict=False)
    cls_model = torch.nn.DataParallel(cls_model).cuda()
    cls_model.eval()

    # ── 3) Prepare transforms
    hrnet_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    cls_tf = transforms.Compose([
        transforms.Resize((120, 184)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])

    # ── 4) Build dataset once (so we can re‑use gt_class_keypoints_dict)
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        hrnet_tf
    )

    # ── 5) Loop per‑image: classify → detect → visualize → save
    for i, entry in enumerate(valid_dataset.db):
        img_path = entry['image']

        # 5.1) classify
        pil = Image.open(img_path).convert('RGB')
        inp = cls_tf(pil).unsqueeze(0).cuda()
        with torch.no_grad():
          logits = cls_model(inp)
        pred = logits.argmax(1).item()      # still 0–8
        category_id = CLS_TO_DET[pred]      # now 1–13, matches your gt_class_keypoints_dict
        print(f"[Image {i}] classifier→detector category {pred}→{category_id}")

        # 5.2) get HRNet input + meta (center, scale)
        hr_input, _, _, meta = valid_dataset[i]
        hr_input = hr_input.unsqueeze(0).cuda()
        c = meta['center']
        s = meta['scale']

        # 5.3) forward through HRNet
        with torch.no_grad():
            output = pose_model(hr_input)

        # output.shape == (1, num_joints, H_out, W_out)
        _, _, H_out, W_out = output.shape
        print("  → actual heatmap shape (H_out, W_out) =",
              H_out, W_out)
        print("  → config.HEATMAP_SIZE             =",
              cfg.MODEL.HEATMAP_SIZE)

        # 5.4) mask out irrelevant heatmap channels
        mask = torch.zeros_like(output).float().cuda()
        rg = valid_dataset.gt_class_keypoints_dict[category_id]
        idxs = torch.arange(rg[0], rg[1], device=mask.device)
        mask[:, idxs, :, :] = 1
        output = output * mask

        preds_local, maxvals = get_final_preds(cfg, output.detach().cpu().numpy(), c, s)

        # Transform back from heatmap coordinate to image coordinate
        preds = preds_local.copy()
        for i in range(preds_local.shape[0]):
            preds[i] = transform_preds(
                preds_local[i], c, s, 
                [cfg.MODEL.HEATMAP_SIZE[0], cfg.MODEL.HEATMAP_SIZE[1]]
            )

        # 5.6) visualize + save
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # preds[0] is a list of (x,y), maxvals[0] is [[conf], [conf], ...]
        for k, (x, y) in enumerate(preds[0]):
            conf = float(maxvals[0][k][0])
            if conf > 0.5:
                cv2.circle(img, (int(x), int(y)), 4, (0,255,0), -1)
                cv2.putText(img, str(k+1), (int(x)+5, int(y)-5),
                            font, 0.5, (0,255,0), 1, cv2.LINE_AA)

        plt.figure(figsize=(12,8))
        plt.imshow(img); plt.axis('off')
        plt.title(f"Cat {category_id} — Img {i}")
        plt.show()

        out_dir = os.path.join(final_output_dir, f"class_{category_id:02d}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"pred_{os.path.basename(img_path)}"
        )
        cv2.imwrite(out_path,
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  → saved to {out_path}")


if __name__ == '__main__':
    main()
