#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

# Import from GarmentIQ
import sys, os
here    = os.path.abspath(os.path.dirname(__file__))
repo    = os.path.abspath(os.path.join(here, os.pardir))
giq_src = os.path.join(repo, 'GarmentIQ', 'src')
if os.path.isdir(giq_src):
    sys.path.insert(0, giq_src)
else:
    raise FileNotFoundError(f"Could not find GarmentIQ/src at {giq_src!r}")
import garmentiq as giq   
from garmentiq.derivation.derive_keypoint_coord import derive_keypoint_coord
from garmentiq.derivation.utils import parse_derivation_args
import garmentiq.refinement
from garmentiq.refinement.refinement import refine_landmark_with_blur

if repo not in sys.path:
    sys.path.insert(0, repo)

import argparse
import pprint
import json

from PIL import Image
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import _init_paths
from config import cfg, update_config
from core.inference import get_final_preds
from utils.transforms import transform_preds
from utils.utils import create_logger
import dataset
import models
from models.tiny_vit import tinyViT


# classifier→detector map
CLS_TO_DET = {
    0:11, 1:2,  2:10, 3:1,
    4:7,  5:9,  6:8,  7:5,
    8:12
}
# classifier idx→label
CLOTHING_LABELS = {
    "0": "long sleeve dress",
    "1": "long sleeve top",
    "2": "short sleeve dress",
    "3":"short sleeve top",
    "4": "shorts",
    "5": "skirt",
    "6": "trousers",
    "7": "vest",
    "8": "vest dress"
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Auto-seg, classify, detect, measure'
    )
    parser.add_argument('--cfg',            required=True, type=str)
    parser.add_argument('--model_file',     required=True, type=str)
    parser.add_argument('--cls_model_file', required=True, type=str)
    parser.add_argument('--data_dir',       default='',    type=str)
    parser.add_argument('--modelDir',      default='',    type=str)
    parser.add_argument('--logDir',        default='',    type=str)
    parser.add_argument('--dataDir',       default='',    type=str)
    parser.add_argument('--prevModelDir',  default='',    type=str)
    parser.add_argument('--save_coords',    default=True, type=bool)
    parser.add_argument('--do_seg',         default=True, type=bool)
    parser.add_argument('--do_refine',         default=False, type=bool)
    parser.add_argument('opts',             nargs=argparse.REMAINDER)
    return parser.parse_args()

def main():
    args = parse_args()
    update_config(cfg, args)

    # override root
    cfg.defrost()
    if args.data_dir:
        cfg.DATASET.ROOT = args.data_dir
    cfg.freeze()

    logger, final_output_dir, _ = create_logger(cfg, args.cfg, 'valid')
    logger.info("Args:\n"   + pprint.pformat(vars(args)))
    logger.info("Config:\n" + pprint.pformat(cfg))

    cudnn.benchmark     = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled       = cfg.CUDNN.ENABLED

    # 1) segmentation
    if args.do_seg:
        logger.info("Loading BiRefNet…")
        BiRefNet = giq.segmentation.load_model(
            pretrained_model='ZhengPeng7/BiRefNet',
            pretrained_model_args={'trust_remote_code':True},
            high_precision=True
        )
        logger.info("Seg ready.")

    # 2) HRNet pose
    pose_model = eval(f"models.{cfg.MODEL.NAME}.get_pose_net")(cfg, is_train=False)
    pose_model.load_state_dict(torch.load(args.model_file), strict=False)
    pose_model = torch.nn.DataParallel(pose_model).cuda()
    pose_model.eval()

    # 3) tinyViT
    cls_model = tinyViT(
        num_classes=cfg.CLASSIFIER.NUM_CLASSES,
        img_size=tuple(cfg.CLASSIFIER.IMG_SIZE),
        patch_size=cfg.CLASSIFIER.PATCH_SIZE
    )
    state = torch.load(args.cls_model_file, map_location='cpu')
    from collections import OrderedDict
    new_state = OrderedDict({k.replace("module.",""):v for k,v in state.items()})
    cls_model.load_state_dict(new_state, strict=False)
    cls_model = torch.nn.DataParallel(cls_model).cuda()
    cls_model.eval()

    # 4) Transforms
    hrnet_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    cls_tf = transforms.Compose([
        transforms.Resize(tuple(cfg.CLASSIFIER.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 5) Dataset
    valid_dataset = eval(f"dataset.{cfg.DATASET.DATASET}")(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, hrnet_tf
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 6) Loop
    for idx, entry in enumerate(valid_dataset.db):
        img_path = entry['image']
        logger.info(f"[{idx}] {img_path}")

        # 6.1) segmentation
        if args.do_seg:
            orig, mask = giq.segmentation.extract(
                model=BiRefNet,
                image_path=img_path,
                resize_dim=(1024,1024),
                normalize_mean=[0.485,0.456,0.406],
                normalize_std=[0.229,0.224,0.225],
                high_precision=True
            )
            # collapse to H×W
            mask_np = mask
            if mask_np.ndim==3:
                if mask_np.shape[2]==1:
                    mask_np=mask_np[:,:,0]
                else:
                    mask_np=mask_np.max(axis=2)
            mask_uint8 = (mask_np).round().astype(np.uint8)
            mask_uint8[mask_uint8>128]=255
            mask_uint8[mask_uint8<=128]=0
            mask_f = (mask_uint8.astype(np.float32)/255.0)[:,:,None]
            masked = (orig * mask_f).clip(0,1)
            # pil = Image.fromarray((masked*255).astype(np.uint8))
            pil = Image.open(img_path).convert("RGB") # with the fine-tuned model we use the original for prediction instead
        else:
            pil = Image.open(img_path).convert("RGB")

        # 6.2) classify
        inp = cls_tf(pil).unsqueeze(0).cuda()
        with torch.no_grad():
            logits = cls_model(inp)
        cls_idx     = logits.argmax(1).item()
        category_id = CLS_TO_DET[cls_idx]
        label       = CLOTHING_LABELS[str(cls_idx)]
        logger.info(f" → class {cls_idx}→{category_id} ({label})")

        # move mask
        if args.do_seg:
            seg_dir = os.path.join(final_output_dir,
                                   f"class_{category_id:02d}",
                                   "segmentation")
            os.makedirs(seg_dir,exist_ok=True)
            m_out = os.path.join(seg_dir,f"mask_{os.path.basename(img_path)}")
            cv2.imwrite(m_out, mask_uint8)
            logger.info(f"   ↳ mask→{m_out}")

        # 6.3) landmark detect
        hr_input, _, _, meta = valid_dataset[idx]
        hr_input=hr_input.unsqueeze(0).cuda()
        c,s=meta['center'],meta['scale']
        with torch.no_grad():
            out=pose_model(hr_input)
        rg=valid_dataset.gt_class_keypoints_dict[category_id]
        out=out[:,rg[0]:rg[1],:,:]
        preds_local,maxvals = get_final_preds(cfg,out.cpu().numpy(),c[None],s[None])
        preds=preds_local.copy()
        for b in range(preds.shape[0]):
            preds[b]=transform_preds(
                preds_local[b],c,s,
                [cfg.MODEL.HEATMAP_SIZE[0],cfg.MODEL.HEATMAP_SIZE[1]]
            )

        # draw + collect coords
        bgr=cv2.imread(img_path)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        coords=[]
        coord_map={}
        
        if args.do_refine:
            # Apply Gaussian blur to the mask to refine the spatial distribution.
            mask = cv2.imread(m_out, cv2.IMREAD_GRAYSCALE)
            blurred_mask = cv2.GaussianBlur(mask, (11, 11), 0)
    
            for j,(x_np,y_np) in enumerate(preds[0]):
                conf=float(maxvals[0][j][0])# assuming first 25 keypoints
                if conf > 0:
                    x = float(x_np)
                    y = float(y_np)
                    # Refine the landmark using the blurred mask.
                    refined_x, refined_y = refine_landmark_with_blur(x, y, blurred_mask, window_size=5)
                    
                    cv2.circle(rgb,(int(refined_x),int(refined_y)),4,(0,255,0),-1)
                    cv2.putText(rgb,str(j+1),(int(refined_x)+5,int(refined_y)-5),
                                font,0.5,(0,255,0),1,cv2.LINE_AA)
                    coords.append({'keypoint_id':j+1,'x':refined_x,'y':refined_y,'confidence':conf})
                    coord_map[str(j+1)]=(refined_x,refined_y,conf)
        else:
            for j,(x_np,y_np) in enumerate(preds[0]):
                conf=float(maxvals[0][j][0])
                if conf>0:
                    x = float(x_np)
                    y = float(y_np)
                    cv2.circle(rgb,(int(x),int(y)),4,(0,255,0),-1)
                    cv2.putText(rgb,str(j+1),(int(x)+5,int(y)-5),
                                font,0.5,(0,255,0),1,cv2.LINE_AA)
                    coords.append({'keypoint_id':j+1,'x':x,'y':y,'confidence':conf})
                    coord_map[str(j+1)]=(x,y,conf)

        # 6.4) save image + coords
        clsf     = f"class_{category_id:02d}"
        img_dir  = os.path.join(final_output_dir,clsf,"images")
        coord_dir= os.path.join(final_output_dir,clsf,"coordinates")
        os.makedirs(img_dir,exist_ok=True)
        os.makedirs(coord_dir,exist_ok=True)

        out_img=os.path.join(img_dir,f"pred_{os.path.basename(img_path)}")
        #cv2.imwrite(out_img,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        #logger.info(f"   ↳ saved image→{out_img}")

        if args.save_coords and coords:
            coords_path=os.path.join(coord_dir,
                f"coords_{os.path.basename(img_path)}.json")
            with open(coords_path,'w') as f:
                json.dump({
                    'image_path':img_path,
                    'category_id':category_id,
                    'coordinates':coords
                },f,indent=2)
            logger.info(f"   ↳ saved coords→{coords_path}")

        # 6.5) measurements
        instr_path=os.path.join(repo,
            "measurement_instructions_inditex",
            f"{label}.json")
        if os.path.isfile(instr_path):
            instr = json.load(open(instr_path))
            info = instr.get(label, {})
            meas_defs = info.get("measurements", {})
            # only use predefined landmarks list
            lm_defs = info.get("landmarks", {})
            meas_res = {}
            
            # compute each measurement
            for mname, mdef in meas_defs.items():
                s_id = mdef["landmarks"]["start"]
                e_id = mdef["landmarks"]["end"]
                lm_s = lm_defs.get(s_id)
                lm_e = lm_defs.get(e_id)
                
                # Handle missing landmark definitions
                if not lm_s or not lm_e:
                    continue
        
                # --- START Landmark ---
                if lm_s.get("predefined"):
                    if s_id not in coord_map:
                        print(f"Missing coords for {s_id}")
                        continue
                    x1, y1, s_conf = coord_map[s_id]
                else:
                    deriv_s = lm_s.get("derivation")
                    if not deriv_s:
                        continue
                    try:
                        args_s = parse_derivation_args(deriv_s, coords_path, m_out)
                        x1, y1 = derive_keypoint_coord(**args_s)
                        conf = "/"  # Mark the confidence as derived
                        cv2.circle(rgb, (int(x1), int(y1)), 4, (0, 255, 0), -1)
                        cv2.putText(rgb, str(s_id), (int(x1) + 5, int(y1) - 5),
                                    font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Failed to derive {s_id}: {e}")
                        continue
                
                # --- END Landmark ---
                if lm_e.get("predefined"):
                    if e_id not in coord_map:
                        print(f"Missing coords for {e_id}")
                        continue
                    x2, y2, e_conf = coord_map[e_id]
                else:
                    deriv_e = lm_e.get("derivation")
                    if not deriv_e:
                        continue
                    try:
                        args_e = parse_derivation_args(deriv_e, coords_path, m_out)
                        x2, y2 = derive_keypoint_coord(**args_e)
                        e_conf = "/"  # Mark the confidence as derived
                        cv2.circle(rgb, (int(x2), int(y2)), 4, (0, 255, 0), -1)
                        cv2.putText(rgb, str(e_id), (int(x2) + 5, int(y2) - 5),
                                    font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Failed to derive {e_id}: {e}")
                        continue

                
                # Compute Euclidean distance
                dist=float(np.hypot(x2-x1,y2-y1))
                meas_res[mname]={
                    "landmarks": {"start":{"id":s_id,"x":x1,"y":y1,"condifence":s_conf},"end":{"id":e_id,"x":x2,"y":y2,"condifence":e_conf}},
                    "description": mdef.get("description",""),
                    "result": dist
                }
                # aspect ratios (skip zero-denominators)
                aspect = {}
                names = list(meas_res.keys())
                for i in names:
                    for j in names:
                        if i == j:
                            continue
                        numerator   = meas_res[i]["result"]
                        denominator = meas_res[j]["result"]
                        # skip if denominator is zero (or extremely small)
                        if denominator == 0 or abs(denominator) < 1e-6:
                            logger.warning(
                                f"Skipping aspect ratio '{i}/{j}' because "
                                f"denominator is zero or too small ({denominator})"
                            )
                            continue
                        aspect[f"{i}/{j}"] = numerator / denominator

            # save measurement JSON
            meas_out=os.path.join(coord_dir,
                f"meas_{os.path.basename(img_path)}.json")
            with open(meas_out,'w') as f:
                json.dump({
                    "image_path":img_path,
                    "category_id":category_id,
                    "measurements":meas_res,
                    "aspect_ratio":aspect
                },f,indent=2)
            logger.info(f"   ↳ saved measurements→{meas_out}")
            
            cv2.imwrite(out_img,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            logger.info(f"   ↳ saved image→{out_img}")

    logger.info("All done!")

if __name__=="__main__":
    main()
