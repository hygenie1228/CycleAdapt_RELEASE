import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json
import copy
import cv2
from torch.utils.data import Dataset

from core.config import cfg
from core.logger import logger
from funcs_utils import load_img, split_into_chunks, batch_rodrigues, axis_angle_to_6d, rotmat_to_6d, transform_joint_to_other_db
from aug_utils import img_processing, coord2D_processing, coord3D_processing, smpl_param_processing, flip_joint, apply_noise
from human_models import smpl


class BaseDataset(Dataset):
    def __init__(self):
        self.transform = None
        self.data_split = None
        self.has_joint_cam = False
        self.has_smpl_param = False
        self.mode = ''

    def __len__(self):
        if self.mode == 'hmr': return len(self.datalist)
        elif self.mode == 'md': return len(self.vid_indices)
        else: return len(self.datalist)

    def split_dataset(self):
        datalist, vid_indices = {}, {}
        for data in self.datalist:
            seq_name = data['seq_name']
            if seq_name not in datalist:
                datalist[seq_name] = [data]
            else:
                datalist[seq_name].append(data)

        for seq_name in datalist.keys():
            vid_indices[seq_name] = split_into_chunks(np.zeros(len(datalist[seq_name])), cfg.MD.seqlen, self.stride)

        self.datalist, self.vid_indices = None, None
        return datalist, vid_indices

    def select_dataset(self, datalist, vid_indices, seq_name):
        hmr_dataset = copy.deepcopy(self)
        hmr_dataset.datalist = datalist[seq_name]
        hmr_dataset.seq_names = [seq_name]
        hmr_dataset.mode = 'hmr'

        md_dataset = copy.deepcopy(self)
        md_dataset.datalist = datalist[seq_name]
        md_dataset.vid_indices = vid_indices[seq_name]
        md_dataset.seq_names = [seq_name]
        md_dataset.mode = 'md'
        return hmr_dataset, md_dataset
  
    def __getitem__(self, index):
        if self.mode == 'hmr':
            return self.get_item_hmr(index)
        elif self.mode == 'md':
            return self.get_item_md(index)

    def get_item_hmr(self, index):
        data = copy.deepcopy(self.datalist[index])
        
        img_path = data['img_path']
        img = load_img(img_path)

        bbox, joint_img = data['bbox'], data['joint_img']
        joint_img, joint_img_valid = joint_img[:, :2], joint_img[:, -1]

        img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, cfg.HMR.input_img_shape, self.data_split)
        joint_img = coord2D_processing(joint_img, img2bb_trans, do_flip, cfg.HMR.input_img_shape, self.joint_set['flip_pairs'])
        if do_flip: joint_img_valid = flip_joint(joint_img_valid, None, self.joint_set['flip_pairs'])
        
        if self.has_joint_cam:
            joint_cam = coord3D_processing(data['joint_cam'][:,:3], rot, do_flip, self.joint_set['flip_pairs'])
            joint_cam = joint_cam - joint_cam[self.root_joint_idx]
            joint_cam_valid = data['joint_cam'][:,-1]
            if do_flip: joint_cam_valid = flip_joint(joint_cam_valid, None, self.joint_set['flip_pairs'])
        else:
            joint_cam = np.zeros((self.joint_set['joint_num'], 3)).astype(np.float32)
            joint_cam_valid = np.zeros((self.joint_set['joint_num'],)).astype(np.float32)
            
        if self.has_smpl_param:
            smpl_pose, smpl_shape = smpl_param_processing(data['smpl_param'], data['cam_param'], do_flip, rot)
            if 'gender' in data['smpl_param']: gender = data['smpl_param']['gender']
            else: gender = 'neutral'
            mesh_cam, smpl_joint_cam = smpl.get_smpl_coord(smpl_pose, smpl_shape, gender=gender)
            smpl_pose = axis_angle_to_6d(torch.tensor(smpl_pose).reshape(-1,3)).numpy().reshape(-1)
            has_param = np.array([1])
        else:
            smpl_pose, smpl_shape = np.zeros((smpl.joint_num*3,)).astype(np.float32), np.zeros((smpl.shape_dim,)).astype(np.float32)
            smpl_joint_cam = np.zeros((smpl.joint_num, 3)).astype(np.float32)
            mesh_cam = np.zeros((smpl.vertex_num, 3)).astype(np.float32)
            has_param = np.array([0])

        img = self.transform(img.astype(np.float32)/255.0)

        if self.data_split == 'train':
            joints = np.concatenate((joint_img, joint_img_valid[:,None], joint_cam, joint_cam_valid[:,None]), 1)
            joints = transform_joint_to_other_db(joints, self.joint_set['joints_name'], smpl.joints_name)
            joint_img, joint_img_valid, joint_cam, joint_cam_valid = joints[:,:2], joints[:,2], joints[:,3:6], joints[:,6]

            inputs = {'img': img}
            targets = {'pose': smpl_pose, 'shape': smpl_shape, 'joint_img': joint_img, 'joint_cam': joint_cam, 'mesh_cam': mesh_cam}
            meta_info = {'joint_img_valid': joint_img_valid, 'seq_name': data['seq_name'], 'index':index, 'ann_id': data['ann_id']}
        else:
            if self.joint_set['name'] in ['3DPW']:
                joint_cam = np.dot(smpl.h36m_joint_regressor, mesh_cam)
                root_cam = joint_cam[smpl.h36m_root_joint_idx]
                joint_cam, mesh_cam = joint_cam - root_cam, mesh_cam - root_cam
                mesh_cam, joint_cam = mesh_cam * 1000, joint_cam * 1000
            else:
                joint_cam = np.zeros((17, 3)).astype(np.float32)
                mesh_cam = np.zeros((smpl.vertex_num, 3)).astype(np.float32)

            inputs = {'img': img}
            targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'mesh_cam': mesh_cam}
            meta_info = {'seq_name': data['seq_name'], 'index':index, 'ann_id': data['ann_id'], 'bbox': bbox, 'img_path': img_path}
        return inputs, targets, meta_info

    def get_item_md(self, index):
        start_index, end_index = self.vid_indices[index]

        poses_noisy, poses_clean, poses_valid = [], [], []
        for i in range(start_index, end_index+1):
            try:
                poses_noisy.append(np.array(self.datalist[i]['smpl_param']['pose']))
                poses_clean.append(np.array(self.datalist[i]['smpl_param']['pose']))
                poses_valid.append(1)
            except:
                poses_noisy.append(np.zeros((72,)))
                poses_clean.append(np.zeros((72,)))
                poses_valid.append(0)
        poses_noisy = np.stack(poses_noisy).reshape(-1, 3).astype(np.float32)
        poses_clean = np.stack(poses_clean).reshape(-1, 3).astype(np.float32)
        poses_valid = np.array(poses_valid).astype(np.float32)
        poses_noisy, poses_clean = torch.tensor(poses_noisy), torch.tensor(poses_clean)

        # noise augmentation
        if (self.data_split=='train') and (cfg.AUG.pose_noise_ratio > 0):
            poses_noisy = apply_noise(poses_noisy, cfg.AUG.pose_noise_ratio)
        
        # to 6D representations
        poses_noisy = axis_angle_to_6d(poses_noisy).reshape(-1, 24*6)
        poses_clean = axis_angle_to_6d(poses_clean).reshape(-1, 24*6)

        # meta info
        try: shape = self.datalist[start_index]['smpl_param']['shape']
        except: shape = np.zeros((10,)).astype(np.float32)
        try: gender = self.datalist[start_index]['smpl_param']['gender']
        except: gender = 'neutral'
            
        indices = np.arange(start_index, end_index+1)
        indices[indices>=len(self.datalist)] = 0    # put dummy value in indices

        mask = np.ones((cfg.MD.seqlen,)).astype(np.float32)
        if self.data_split == 'train':
            mask_idx = np.delete(np.arange(cfg.MD.seqlen), cfg.MD.mid_frame)
            if cfg.MD.mid_frame % 2 == 0: mask_idx = np.random.choice(mask_idx, cfg.MD.seqlen//2, replace=False)
            else: mask_idx = np.random.choice(mask_idx, cfg.MD.seqlen//2-1, replace=False)
            mask_idx = np.append(mask_idx, cfg.MD.mid_frame)
        else:
            if cfg.MD.mid_frame % 2 == 0: mask_idx = np.arange(0,cfg.MD.seqlen,2)
            else: mask_idx = np.arange(1,cfg.MD.seqlen,2)
        mask[mask_idx] = 0

        inputs = {'poses': poses_noisy}
        targets = {'poses': poses_clean}
        meta_info = {'mask': mask, 'valid': poses_valid, 'shape': shape, 'gender':gender, 'indices': indices}
        return inputs, targets, meta_info