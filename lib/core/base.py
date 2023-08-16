import sys
import os
import os.path as osp
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import cfg
from core.logger import logger

from dataset.multiple_datasets import MultipleDatasets
from funcs_utils import axis_angle_to_6d, rot6d_to_axis_angle, batch_rodrigues, slide_window_to_sequence
from train_utils import load_checkpoint, count_parameters, AverageMeterDict
from eval_utils import eval_hmr
from human_models import smpl

for dataset in cfg.DATASET.train_list+cfg.DATASET.test_list:
    exec(f'from {dataset}.dataset import {dataset}')

class Adaptater:
    def __init__(self, is_adapt=True):
        if is_adapt:
            self.hmr_datasets, self.md_datasets, self.hmr_dataloaders, self.md_dataloaders = get_dataloader(cfg.DATASET.test_list, is_train=True)
            _, _, self.hmr_test_dataloaders, self.md_test_dataloaders = get_dataloader(cfg.DATASET.test_list, is_train=False)
        else:
            self.hmr_datasets, self.md_datasets, self.hmr_test_dataloaders, self.md_test_dataloaders = get_dataloader(cfg.DATASET.test_list, is_train=False)
        
        self.smpl_layer = copy.deepcopy(smpl.layer['neutral']).cuda()
        self.J_regressor = torch.from_numpy(smpl.h36m_joint_regressor).float().cuda()
        
        self.is_adapt = is_adapt
        self.video_num = len(self.hmr_datasets)
        self.hmr_dict, self.md_dict = None, None
        self.dict_length = None 

        if not is_adapt: 
            self.run = self.preprocess
    
    def preprocess(self, args, model_path, video_idx):
        self.seq_name = self.hmr_datasets[video_idx].seq_names[0]
        self.video_len = len(self.hmr_datasets[video_idx])
        
        if not self.is_adapt:
            model_path = osp.join(model_path, 'checkpoints', f'{self.seq_name}.pth.tar')
            if not osp.exists(model_path): return
        self.model, checkpoint = prepare_network(args, model_path, self.is_adapt)

        if self.is_adapt: 
            self.loss, optimizer, lr_scheduler = adapt_setup(self.model, checkpoint)
            self.hmr_optimizer, self.md_optimizer = optimizer
            self.hmr_lr_scheduler, self.md_lr_scheduler = lr_scheduler
        
        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model)
        self.eval_history = {'mpjpe': [], 'pa-mpjpe': [], 'mpvpe': []}
        self.eval_vals = {}

        if video_idx == 0:
            if model_path and isinstance(model_path, tuple):
                logger.info(f"==> Loading HMRNet checkpoint: {model_path[0]}")
                logger.info(f"==> Loading MDNet checkpoint: {model_path[1]}")
            else:
                logger.info(f"==> Loading from checkpoint: {model_path}")
            logger.info(f"# of model parameters: {count_parameters(self.model)}")        

        logger.info(f'==> ({video_idx+1}/{self.video_num}) video name: {self.seq_name}')
        self.dict_length = len(self.hmr_datasets[video_idx])
        self.hmr_dict = torch.zeros((self.dict_length,85+87+1))  # last one is validation flag
        self.md_dict = torch.zeros((self.dict_length, 85+87+1))  # last one is validation flag

        # evaluation
        eval_dict = self.evaluation(self.hmr_test_dataloaders[video_idx])
        for k in eval_dict.keys():
            self.eval_vals[k] = np.array(eval_dict[k]).mean()
        self.save_history(video_idx, 0)
        
    def run(self, video_idx, cycle):
        hmr_dataset, hmr_dataloader = self.hmr_datasets[video_idx], self.hmr_dataloaders[video_idx]
        hmr_test_dataloader = self.hmr_test_dataloaders[video_idx]  

        md_dataset, md_dataloader = self.md_datasets[video_idx], self.md_dataloaders[video_idx]
        md_test_dataloader = self.md_test_dataloaders[video_idx]

        self.hmrnet_adaptation_stage(hmr_dataset, hmr_dataloader, cycle)
        self.hmr_lr_scheduler.step()

        self.mdnet_adaptation_stage(md_dataset, md_dataloader, cycle)
        self.md_lr_scheduler.step()

        eval_dict = self.evaluation(hmr_test_dataloader)
        for k in eval_dict.keys():
            self.eval_vals[k] = np.array(eval_dict[k]).mean()
        self.save_history(video_idx, cycle)

        if cfg.TEST.vis and cycle == cfg.TRAIN.total_cycle:
           self.visualization(hmr_test_dataloader, video_idx=video_idx)

    def hmrnet_adaptation_stage(self, dataset, dataloader, cycle):
        self.model.train()
        dataloader = tqdm(dataloader)
        dataloader.set_description(f'(cycle{cycle}) HMRNet adaptation')
        for i, (inputs, targets, meta_info) in enumerate(dataloader):
            db_index = meta_info['index']
            batch_size = db_index.shape[0]

            # fetch mp results and check validation
            mdnet_results = self.md_dict[db_index.reshape(-1)].clone().reshape(batch_size, -1)
            targets['md_pose'] = batch_rodrigues(mdnet_results[:,:72].reshape(-1, 3)).reshape(-1, 24*9)
            targets['md_shape'] = mdnet_results[:,72:72+10]
            targets['md_joint_cam'] = mdnet_results[:,85:85+87].reshape(-1,smpl.joint_num,3)
            meta_info['md_valid'] = mdnet_results[:,-1].bool()

            # fine-tuning
            self.hmr_optimizer.zero_grad()
            output = self.model(inputs, 'hmr')

            # loss calculation
            loss = 0.0
            loss += cfg.TRAIN.pose_loss_weight * self.loss['param'](output['rotmat'], targets['md_pose'], meta_info['md_valid'])                
            loss += cfg.TRAIN.joint_loss_weight * self.loss['joint'](output['joint_cam'], targets['md_joint_cam'], meta_info['md_valid'][:,None].repeat(1,smpl.joint_num))
            loss += cfg.TRAIN.shape_loss_weight * self.loss['param'](output['shape'], targets['md_shape'], meta_info['md_valid'])
            loss += cfg.TRAIN.proj_loss_weight * self.loss['proj'](output['joint_proj'], targets['joint_img'], meta_info['joint_img_valid'])
            loss += cfg.TRAIN.prior_loss_weight * self.loss['prior'](output['pose'][:,3:], output['shape'])
            loss.backward()
            self.hmr_optimizer.step()

            for k,v in output.items():
                if isinstance(v, torch.Tensor):
                    output[k] = v.detach()

            # save results in dictionary
            save_theta = torch.cat((output['pose'], output['shape'], output['trans']), dim=-1).cpu()
            save_joint_cam = output['joint_cam'].cpu().reshape(batch_size,-1)
            results = torch.cat((save_theta, save_joint_cam), dim=-1).cpu()
            results = torch.cat((results, torch.ones(len(results), 1)), dim=-1)
            self.hmr_dict[db_index] = results


    def mdnet_adaptation_stage(self, dataset, dataloader, cycle):
        self.model.train()
        dataloader = tqdm(dataloader)
        dataloader.set_description(f'(cycle{cycle}) MDNet adaptation')

        gender = 'neutral'
        md_dict = torch.zeros((self.dict_length-cfg.MD.seqlen+1, cfg.MD.seqlen, 144+10+3))
        for i, (inputs, targets, meta_info) in enumerate(dataloader):
            db_indices = meta_info['indices']
            batch_size, seqlen = db_indices.shape[0], db_indices.shape[1]

            # fetch hmr results and check validation            
            hmrnet_results = self.hmr_dict[db_indices.reshape(-1)].clone().reshape(batch_size*seqlen, -1)
            inputs['poses'] = hmrnet_results[..., :72]
            inputs['poses'] = axis_angle_to_6d(inputs['poses']).reshape(batch_size, seqlen, -1)
            targets['hmr_poses'] = inputs['poses'].clone()

            # fine-tuning
            self.md_optimizer.zero_grad()
            output = self.model(inputs, 'md', meta_info)
            
            # loss calculation
            loss = 0.0
            loss += cfg.TRAIN.md_loss_weight * self.loss['param'](output['poses'], targets['hmr_poses'], (1-meta_info['mask']))
            loss.backward()
            self.md_optimizer.step()

            for k,v in output.items():
                if isinstance(v, torch.Tensor):
                    output[k] = v.detach()

            # post processing
            db_indices = db_indices[:,0]
            gender = meta_info['gender'][0]
            save_pose = output['poses'].detach().cpu()
            save_shape = hmrnet_results[:, 72:72+10].reshape(batch_size, seqlen, -1).mean(1)
            save_shape = save_shape[:,None,:].repeat(1,cfg.MD.seqlen,1)
            save_trans = hmrnet_results[:, 82:85].reshape(batch_size, seqlen, -1)
            md_dict[db_indices] = torch.cat((save_pose, save_shape, save_trans), dim=-1)

        self.mdnet_postprocessing(md_dict, gender, save_result=True)


    def evaluation(self, dataloader, save_result=False):
        eval_metrics = list(self.eval_history.keys())
        pred_joint_cam_results, gt_joint_cam_results, pred_mesh_cam_results, gt_mesh_cam_results = [], [], [], []
        self.model.eval()

        for i, (inputs, targets, meta_info) in enumerate(dataloader):
            db_index = meta_info['index']
            batch_size = db_index.shape[0]

            with torch.no_grad():
                output = self.model(inputs, 'hmr')
            
            pred_joint_cam, tar_joint_cam, pred_mesh_cam, tar_mesh_cam = self.hmrnet_postprocessing(output, targets, meta_info)
            pred_joint_cam_results.append(pred_joint_cam);  gt_joint_cam_results.append(tar_joint_cam)
            pred_mesh_cam_results.append(pred_mesh_cam); gt_mesh_cam_results.append(tar_mesh_cam)
            
            # save results in dictionary
            if save_result:
                save_theta = torch.cat((output['pose'], output['shape'], output['trans']), dim=-1).cpu()
                save_joint_cam = output['joint_cam'].cpu().reshape(batch_size,-1)
                results = torch.cat((save_theta, save_joint_cam), dim=-1)
                results = torch.cat((results, torch.ones(len(results), 1)), dim=-1)
                self.hmr_dict[db_index] = results

        pred_joint_cam_results = np.concatenate(pred_joint_cam_results); gt_joint_cam_results = np.concatenate(gt_joint_cam_results)
        pred_mesh_cam_results = np.concatenate(pred_mesh_cam_results); gt_mesh_cam_results = np.concatenate(gt_mesh_cam_results)
        eval_dict = eval_hmr(pred_joint_cam_results, gt_joint_cam_results, pred_mesh_cam_results, gt_mesh_cam_results, metrics=eval_metrics)
        return eval_dict

    def save_history(self, video_idx, cycle):   
        for k in self.eval_history.keys():
            self.eval_history[k].append(self.eval_vals[k])

    def visualization(self, dataloader, video_idx=0, cycle=0):
        from vis_utils import render_mesh, get_color, save_obj
        from funcs_utils import convert_focal_princpt
        self.model.eval()

        dataloader = tqdm(dataloader)
        dataloader.set_description('Rendering')
        for i, (inputs, targets, meta_info) in enumerate(dataloader):
            db_index = meta_info['index']
            batch_size = db_index.shape[0]

            if True: results = self.hmr_dict[db_index.reshape(-1)].clone().reshape(batch_size, -1)
            else: results = self.md_dict[db_index.reshape(-1)].clone().reshape(batch_size, -1)
            pose = results[:,:72].cuda()
            shape = results[:,72:72+10].cuda()
            trans = results[:,82:85].cuda()

            output = self.smpl_layer(betas=shape, body_pose=pose[:,3:], global_orient=pose[:,:3])
            pred_mesh_cam = output.vertices + trans[:,None,:]
            pred_mesh_cam = pred_mesh_cam.cpu().numpy()

            for j in range(batch_size):
                focal, princpt = convert_focal_princpt(meta_info['bbox'][j])
                cam_param = {'focal': focal, 'princpt': princpt}
                img = cv2.imread(meta_info['img_path'][j])
                img, rgba = render_mesh(img, pred_mesh_cam[j], smpl.face, cam_param, color=np.array([190,190,190])/255, return_mesh=True)
                
                os.makedirs(osp.join(cfg.vis_dir, 'tmp', f'{str(video_idx)}'), exist_ok=True)
                cv2.imwrite(osp.join(cfg.vis_dir, f'tmp', f'{str(video_idx)}', meta_info['img_path'][j].split('/')[-1]), rgba)
                
    def hmrnet_postprocessing(self, output, targets, meta_info=None):
        pred_mesh_cam = output['mesh_cam']
        pred_mesh_cam = pred_mesh_cam * 1000
        pred_joint_cam = torch.matmul(self.J_regressor[None, :, :], pred_mesh_cam)
        pred_mesh_cam = pred_mesh_cam - pred_joint_cam[:, [0]]
        pred_joint_cam = pred_joint_cam - pred_joint_cam[:, [0]]

        pred_mesh_cam = pred_mesh_cam.detach().cpu().numpy()
        tar_mesh_cam = targets['mesh_cam'].cpu().numpy() 
        pred_joint_cam = pred_joint_cam.detach().cpu().numpy()

        if targets['joint_cam'].shape[1] > 24:
            tar_mesh_cam = targets['mesh_cam'].cuda() * 1000
            tar_joint_cam = torch.matmul(self.J_regressor[None, :, :], tar_mesh_cam)
            tar_mesh_cam, tar_joint_cam = tar_mesh_cam - tar_joint_cam[:, [0]], tar_joint_cam - tar_joint_cam[:, [0]]
            tar_mesh_cam, tar_joint_cam = tar_mesh_cam.cpu().numpy(), tar_joint_cam.cpu().numpy()
        else:
            tar_joint_cam = targets['joint_cam'].cpu().numpy()  
        return pred_joint_cam, tar_joint_cam, pred_mesh_cam, tar_mesh_cam

    def mdnet_postprocessing(self, md_dict, gender='neutral', save_result=False):
        md_dict = slide_window_to_sequence(md_dict)
        total_len = len(self.md_dict)
        batch_size = cfg.TEST.batch_size
        for i in range(int(np.ceil(total_len/batch_size))):
            if i == total_len//batch_size:
                indices = torch.arange(batch_size*i, total_len)
            else:
                indices = torch.arange(batch_size*i, batch_size*(i+1))
            
            mdnet_results = md_dict[indices]
            pose = mdnet_results[:, :144].cuda().reshape(-1, 6)
            pose = rot6d_to_axis_angle(pose).reshape(-1, 72)
            shape = mdnet_results[:, 144:144+10].cuda().reshape(-1, 10)
            trans = mdnet_results[:, 144+10:].cuda().reshape(-1, 3)

            output = self.smpl_layer(betas=shape, body_pose=pose[:,3:], global_orient=pose[:,:3])
            pred_mesh_cam = output.vertices

            # save results in dictionary
            if save_result:
                save_theta = torch.cat((pose, shape, trans), dim=-1)
                joint_cam = torch.matmul(torch.from_numpy(smpl.joint_regressor)[None,:,:].cuda(), pred_mesh_cam)
                save_joint_cam = (joint_cam - joint_cam[:,[0]]).reshape(-1,87)    
                results = torch.cat((save_theta, save_joint_cam), dim=-1).cpu()
                results = torch.cat((results, torch.ones(len(results), 1)), dim=-1)
                self.md_dict[indices] = results

    
def get_dataloader(dataset_names, is_train):
    from train_utils import worker_init_fn
    if len(dataset_names) == 0: assert 0
    dataset_split = 'TRAIN' if is_train else 'TEST'   
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    for name in dataset_names:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        dataset = eval(f'{name}')(transform, dataset_split.lower())
        dataset_list.append(dataset)
        if not is_train:
            logger.info(f"# of test {name} data: {len(dataset)}")

    # Construct DataLoader
    dataset = dataset_list[0]
    datalist, vid_indices = dataset.split_dataset()

    if len(cfg.DATASET.select_seq_name) != 0: target_seq_names = cfg.DATASET.select_seq_name
    else: target_seq_names = dataset.seq_names

    hmr_dataset_list, md_dataset_list, hmr_dataloader_list, md_dataloader_list = [], [], [], []
    for name in target_seq_names:
        hmr_dataset, md_datatset = dataset.select_dataset(datalist, vid_indices, name)
        
        # hmr dataset
        hmr_dataset_list.append(hmr_dataset)
        dataloader = DataLoader(hmr_dataset, batch_size=batch_per_dataset, shuffle=cfg[dataset_split].shuffle, 
                                num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        hmr_dataloader_list.append(dataloader)
        
        # md dataset
        md_dataset_list.append(md_datatset)
        dataloader = DataLoader(md_datatset, batch_size=batch_per_dataset, shuffle=cfg[dataset_split].shuffle, 
                                num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        md_dataloader_list.append(dataloader)

    return hmr_dataset_list, md_dataset_list, hmr_dataloader_list, md_dataloader_list

def prepare_network(args, load_dir='', is_train=True): 
    from models.model import get_model  
    model, checkpoint = None, None
    model = get_model(is_train)

    if load_dir and isinstance(load_dir, tuple):
        checkpoint_hmr = load_checkpoint(load_dir=load_dir[0])
        checkpoint_mp = load_checkpoint(load_dir=load_dir[1])
        checkpoint = (checkpoint_hmr, checkpoint_mp)
        model.load_weights(checkpoint)
    elif load_dir and (not is_train or args.resume_training):
        checkpoint = load_checkpoint(load_dir=load_dir)
        model.load_weights(checkpoint)
    return model, checkpoint

def adapt_setup(model, checkpoint):    
    from core.loss import get_loss
    from train_utils import get_optimizer, get_scheduler
    criterion = get_loss()

    hmr_optimizer = get_optimizer(model=model, mode='hmr')
    hmr_lr_scheduler = get_scheduler(optimizer=hmr_optimizer)

    for state in hmr_optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    md_optimizer = get_optimizer(model=model, mode='md')
    md_lr_scheduler = get_scheduler(optimizer=md_optimizer)
    
    for state in md_optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    optimizer = (hmr_optimizer, md_optimizer)
    lr_scheduler = (hmr_lr_scheduler, md_lr_scheduler)
    return criterion, optimizer, lr_scheduler