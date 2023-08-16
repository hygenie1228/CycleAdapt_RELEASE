import os
import os.path as osp
import argparse
import subprocess
import torch
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import __init_path

from core.config import cfg, update_config
from core.logger import logger

from train_utils import save_checkpoint, check_data_parallel
from vis_utils import print_eval_history

parser = argparse.ArgumentParser(description='Run CycleAdapt')
parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
parser.add_argument('--cfg', type=str, help='experiment configure file name')
parser.add_argument('--resume_training', action='store_true', help='resume training')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path for resume_training')

args = parser.parse_args()
if args.cfg: update_config(args.cfg)

np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
logger.info(f"Work on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Args: {args}")
logger.info(f"Cfg: {cfg}")

if args.checkpoint != '': model_path = args.checkpoint
else: model_path = (cfg.HMR.weight_path, cfg.MD.weight_path)

from core.base import Adaptater
adaptater = Adaptater()

total_eval_history = {}
total_video_len = 0
logger.info(f"===> Start Adapation...")
for video_idx in tqdm(range(adaptater.video_num)):
    adaptater.preprocess(args, model_path, video_idx)
    
    for cycle in range(cfg.TRAIN.total_cycle + 1):
        adaptater.run(video_idx, cycle)      

    save_checkpoint({
        'cycle': cycle,
        'model_state_dict': check_data_parallel(adaptater.model.state_dict()),
        'optim_state_dict': (adaptater.hmr_optimizer.state_dict(), adaptater.md_optimizer.state_dict()),
        'scheduler_state_dict': (adaptater.hmr_lr_scheduler.state_dict(), adaptater.md_lr_scheduler.state_dict()),
        'test_log': adaptater.eval_history
    }, cycle, file_path=osp.join(cfg.checkpoint_dir, f'{adaptater.seq_name}.pth.tar'))

    if cfg.TEST.vis:
        continue
    
    print_eval_history(adaptater.eval_history)
    prev_total_video_len = total_video_len
    total_video_len += adaptater.video_len

    for k in adaptater.eval_history.keys():
        eval_list = np.array(adaptater.eval_history[k])
        if k not in total_eval_history:
            total_eval_history[k] = eval_list
        else:
            total_eval_history[k] = (total_eval_history[k] * prev_total_video_len + eval_list * adaptater.video_len) / total_video_len

if not (cfg.TEST.vis and cfg.DATASET.test_list[0] == 'Demo'):
    logger.info("\nTotal Evalation Results:")
    print_eval_history(total_eval_history)
else:
    img_list = sorted(glob(osp.join(adaptater.hmr_datasets[0].img_dir, '*.png'))) 
    mesh_list = sorted(glob(osp.join(cfg.vis_dir, 'tmp', '*'))) 
    
    tmp_path = osp.join(cfg.vis_dir, 'tmp')
    out_path = osp.join(tmp_path, 'total')
    os.makedirs(out_path, exist_ok=True)
    logger.info("Total Rendering...")
    for img_path in tqdm(img_list):
        img = cv2.imread(img_path)
        for mesh_path in mesh_list:
            mesh_path = osp.join(mesh_path, img_path.split('/')[-1])
            if not osp.exists(mesh_path): continue

            mesh = cv2.imread(mesh_path, cv2.IMREAD_UNCHANGED)
            mesh, mask = mesh[:,:,:3], (mesh[:,:,[-1]] > 0)
            img = mesh * mask + img * (1 - mask)
        cv2.imwrite(osp.join(out_path, img_path.split('/')[-1]), img)

    command = ['ffmpeg', '-y', '-framerate', '10', '-r', '18', '-i', f'{out_path}/%06d.png', osp.join(cfg.vis_dir, f'result.mp4')]
    subprocess.call(command)
    os.system(f'rm -rf {tmp_path}')
    logger.info("Finished.")