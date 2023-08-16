import os
import torch
import random
import math
import numpy as np
import os.path as osp
import glob
import json
import copy

from core.config import cfg
from funcs_utils import process_bbox, world2cam, cam2pixel, split_into_chunks, transform_joint_to_other_db
from dataset.base_dataset import BaseDataset


class Demo(BaseDataset):
    def __init__(self, transform, data_split, target_seq_name=None):
        super(Demo, self).__init__()
        self.transform = transform
        self.data_split = data_split

        self.img_dir = osp.join('data', 'Demo', 'images')
        self.annot_path = osp.join('data', 'Demo', 'annotation.json')

        self.joint_set = {
            'name': 'MSCOCO',
            'joint_num': 20,
            'joints_name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck', 'Head'),
            'flip_pairs': ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
            'skeleton': ((0, 19), (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 19))
        }
        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis')
        
        self.has_joint_cam = False
        self.has_smpl_param = False
        self.datalist, self.vid_indices = self.load_data()

    def load_data(self):        
        with open(self.annot_path, 'r') as f:
            db = json.load(f)
        
        datalist = {}
        seq_names = []
        for aid in range(len(db)):
            ann = db[aid]
            img_path = osp.join(self.img_dir, ann['image_id'])

            bbox = process_bbox(ann['box'], (0,0), cfg.HMR.input_img_shape, expand_ratio=cfg.DATASET.bbox_expand_ratio) 
            if bbox is None: continue
            
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
            if (joint_img[:, 2] > 0.3).sum() == 0: continue            
            height = self.get_person_height(joint_img)
            score = ann['score']
            if score < 2.5 or height < 250:
                continue

            joint_img_valid = (joint_img[:,[-1]] > 0.3).astype(dtype=np.float32)
            joint_img = joint_img[:,:2]
            joint_img, joint_img_valid = self.add_pelvis_neck_head(joint_img, joint_img_valid, self.joint_set['joints_name'])
            joint_img = np.concatenate((joint_img, joint_img_valid[:,None]), axis=-1).astype(np.float32)
            seq_name = str(int(abs(ann['idx'])))

            data = {
                'ann_id': aid,
                'img_path': img_path,
                'bbox': bbox,
                'joint_img': joint_img, 
                'seq_name': seq_name
            }

            if seq_name not in datalist:
                datalist[seq_name] = [data]
                seq_names.append(seq_name)
            else:
                datalist[seq_name].append(data)

        max_len = 0
        for idx in seq_names:
            if len(datalist[idx]) > max_len:
                max_len = len(datalist[idx]) 
        if max_len < 128: len_thres = max_len
        else: len_thres = 128

        human_idxs = []
        for idx in seq_names:
            if len(datalist[idx]) < len_thres:
                continue    # remove too short clip
            human_idxs.append(idx)

        new_datalist, new_seq_names = [], []
        curr_seq_name = '0'
        for idx in human_idxs:
            for ann in datalist[idx]:
                ann = copy.deepcopy(ann)
                ann['seq_name'] = curr_seq_name
                new_datalist.append(ann)
                new_seq_names.append(curr_seq_name)
            curr_seq_name = str(int(curr_seq_name)+1)
        datalist, seq_names = new_datalist, new_seq_names

        self.stride = 1  
        self.seq_names = np.unique(seq_names)
        vid_indices = split_into_chunks(np.array(seq_names), cfg.MD.seqlen, self.stride)
        return datalist, vid_indices

    def get_person_height(self, j2d):
        vis = j2d[:, 2] > 0.3
        min_j = np.min(j2d[vis,:2], 0)
        max_j = np.max(j2d[vis,:2], 0)
        person_height = np.linalg.norm(max_j - min_j)
        return person_height

    def add_pelvis_neck_head(self, joint_coord, joint_valid, joints_name):
        lhip_idx = joints_name.index('L_Hip')
        rhip_idx = joints_name.index('R_Hip')
        
        if joint_valid[lhip_idx] > 0 and joint_valid[rhip_idx] > 0:
            pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
            pelvis = pelvis.reshape((1, -1))
            joint_valid = np.append(joint_valid, 1)
        else:
            pelvis = np.zeros_like(joint_coord[0, None, :])
            joint_valid = np.append(joint_valid, 0)

        lshoulder_idx = joints_name.index('L_Shoulder')
        rshoulder_idx = joints_name.index('R_Shoulder')

        if joint_valid[lshoulder_idx] > 0 and joint_valid[rshoulder_idx] > 0:
            neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
            neck = neck.reshape((1,-1))
            joint_valid = np.append(joint_valid, 1)
        else:
            neck = np.zeros_like(joint_coord[0, None, :])
            joint_valid = np.append(joint_valid, 0)

        lear_idx = joints_name.index('L_Ear')
        rear_idx = joints_name.index('R_Ear')

        if joint_valid[lear_idx] > 0 and joint_valid[rear_idx] > 0:
            head = (joint_coord[lear_idx, :] + joint_coord[rear_idx, :]) * 0.5
            head = head.reshape((1,-1))
            joint_valid = np.append(joint_valid, 1)
        else:
            head = np.zeros_like(joint_coord[0, None, :])
            joint_valid = np.append(joint_valid, 0)

        joint_coord = np.concatenate((joint_coord, pelvis, neck, head))
        return joint_coord, joint_valid