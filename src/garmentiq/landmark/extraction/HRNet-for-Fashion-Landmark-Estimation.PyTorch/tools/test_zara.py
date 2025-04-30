from __future__ import absolute_import, division, print_function

import argparse
import os
import pprint
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on Zara dataset')
    parser.add_argument('--cfg', required=True, type=str)
    parser.add_argument('--model_file', required=True, type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='previous model directory', type=str, default='')

    return parser.parse_args()


def main():
    args = parse_args()
    update_config(cfg, args)

    # make config mutable
    cfg.defrost()
    if args.data_dir:
        cfg.DATASET.ROOT = args.data_dir

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    model.load_state_dict(torch.load(args.model_file), strict=False)
    model = torch.nn.DataParallel(model).cuda()

    criterion = JointsMSELoss(
        cfg=cfg,
        target_type=cfg.MODEL.TARGET_TYPE,
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([transforms.ToTensor(), normalize])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    logger.info("Running inference...")
    preds, _, _, _ = validate(cfg, valid_loader, valid_dataset, model, criterion, final_output_dir, tb_log_dir, return_output=True)

    for i, pred in enumerate(preds):
      img_path = valid_dataset.db[i]['image']
      img = cv2.imread(img_path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      font = cv2.FONT_HERSHEY_SIMPLEX
      # for x, y, conf in pred:
      #     if conf > 0.5:
      #         cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
      for idx, (x, y, conf) in enumerate(pred[:]):  
        if conf > 0.5:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(img, str(idx + 1), (int(x) + 5, int(y) - 5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

      plt.figure(figsize=(12, 8))
      plt.imshow(img)
      plt.axis('off')
      plt.title(f"Predicted Landmarks - Image {i + 1}")
      plt.show()

      # Save output image
      output_path = os.path.join(final_output_dir, f"pred_{os.path.basename(img_path)}")
      cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      print(f"Saved prediction to {output_path}")

    
if __name__ == '__main__':
    main()