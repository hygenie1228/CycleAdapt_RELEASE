import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from core.config import cfg


class AverageMeterDict(object):
    def __init__(self, names):
        for name in names:
            value = AverageMeter()
            setattr(self, name, value)

    def __getitem__(self,key):
        return getattr(self, key)

    def update(self, name, val, n=1):
        getattr(self, name).update(val, n)

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

def lr_check(optimizer, epoch):
    base_epoch = 5
    if False and epoch <= base_epoch:
        lr_warmup(optimizer, cfg.TRAIN.lr, epoch, base_epoch)

    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
    print(f"Current epoch {epoch}, lr: {curr_lr}")


def lr_warmup(optimizer, lr, epoch, base):
    lr = lr * (epoch / base)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        self.acc += time.time() - self.t0  # cacluate time diff

    def reset(self):
        self.acc = 0

    def print(self):
        return round(self.acc, 2)

def stop():
    sys.exit()


def check_data_parallel(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model, mode=None):
    total_params = []
    
    if mode == 'md':
        total_params += list(model.md_net.parameters())
    elif mode == 'hmr':
        total_params += list(model.hmr_net.parameters())
    else:
        if hasattr(model, 'trainable_modules'):
            for module in model.trainable_modules:
                total_params += list(module.parameters())
        else:
            total_params += list(model.parameters())

    optimizer = None
    if cfg.TRAIN.optimizer == 'sgd':
        optimizer = optim.SGD(
            total_params,
            lr=cfg.TRAIN.lr,
            momentum=cfg.TRAIN.momentum,
            weight_decay=cfg.TRAIN.weight_decay
        )
    elif cfg.TRAIN.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            total_params,
            lr=cfg.TRAIN.lr
        )
    elif cfg.TRAIN.optimizer == 'adam':
        optimizer = optim.Adam(
            total_params,
            lr=cfg.TRAIN.lr,
            betas=(cfg.TRAIN.beta1, cfg.TRAIN.beta2)
        )
    elif cfg.TRAIN.optimizer == 'adamw':
        optimizer = optim.AdamW(
            total_params,
            lr=cfg.TRAIN.lr,
            weight_decay=cfg.TRAIN.weight_decay
        )

    return optimizer


def get_scheduler(optimizer):
    scheduler = None
    if cfg.TRAIN.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.lr_step, gamma=cfg.TRAIN.lr_factor)
    elif cfg.TRAIN.scheduler == 'platue':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.TRAIN.lr_factor, patience=10, min_lr=1e-5)
    elif cfg.TRAIN.scheduler == 'cosine':
        from warmup_scheduler import GradualWarmupScheduler
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.total_cycle-3, eta_min=cfg.TRAIN.min_lr)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_cosine)
        optimizer.zero_grad(); optimizer.step()
        scheduler.step()

    return scheduler


def save_checkpoint(states, epoch, file_path=None, is_best=None):
    if file_path is None:
        file_name = f'epoch_{epoch}.pth.tar'
        output_dir = cfg.checkpoint_dir
        if states['epoch'] == cfg.TRAIN.total_cycle:
            file_name = 'final.pth.tar'
        file_path = os.path.join(output_dir, file_name)
            
    torch.save(states, file_path)

    if is_best:
        torch.save(states, os.path.join(output_dir, 'best.pth.tar'))


def load_checkpoint(load_dir, epoch=0, pick_best=False):
    try:
        checkpoint = torch.load(load_dir, map_location='cuda')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint exists!\n", e)

def worker_init_fn(worder_id):
        np.random.seed(np.random.get_state()[1][0] + worder_id)