import os
import os.path as osp
import argparse
import numpy as np
import torch
import shutil
from tqdm import tqdm
import __init_path

from core.config import update_config, cfg
from core.logger import logger

from vis_utils import print_eval_history

parser = argparse.ArgumentParser(description='Test adapted HMRNet')
parser.add_argument('--cfg', type=str, help='experiment configure file name')
parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
parser.add_argument('--exp-dir', type=str, default='', help='experiment directory path for evaluation')

args = parser.parse_args()
if args.cfg: update_config(args.cfg)
    
np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
logger.info(f"Work on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Args: {args}")
logger.info(f"Cfg: {cfg}")

from core.base import Adaptater
tester = Adaptater(is_adapt=False)

print("===> Start Evaluation...")
total_eval_history, total_video_len = {}, 0
for video_idx in tqdm(range(tester.video_num)):
    tester.run(args, args.exp_dir, video_idx)

    print_eval_history(tester.eval_history)
    prev_total_video_len = total_video_len
    total_video_len += tester.video_len

    for k in tester.eval_history.keys():
        eval_list = np.array(tester.eval_history[k])
        if k not in total_eval_history:
            total_eval_history[k] = eval_list
        else:
            total_eval_history[k] = (total_eval_history[k] * prev_total_video_len + eval_list * tester.video_len) / total_video_len

logger.info("\nTotal Evalation Results:")
print_eval_history(total_eval_history)

