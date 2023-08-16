import os
import torch
import random
import math
import numpy as np
import os.path as osp
import glob
import json
from pycocotools.coco import COCO

from core.config import cfg
from funcs_utils import process_bbox, world2cam, cam2pixel, split_into_chunks, transform_joint_to_other_db
from dataset.base_dataset import BaseDataset


class PW3D(BaseDataset):
    def __init__(self, transform, data_split, target_seq_name=None):
        super(PW3D, self).__init__()
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('data', 'PW3D', 'imageFiles')
        self.annot_path = osp.join('data', 'PW3D', '3DPW_test.json')

        self.joint_set = {
            'name': '3DPW',
            'joint_num': 24,
            'joints_name': ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'),
            'flip_pairs': ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)),
            'skeleton': ((0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        }
        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis')
        self.openpose_joints_name = ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear',  'L_Ear')

        self.has_joint_cam = True
        self.has_smpl_param = True
        self.stride = 1
        self.datalist, self.vid_indices = self.load_data()

    def load_data(self):            
        db = COCO(osp.join(self.annot_path))

        datalist, seq_names = [], []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['sequence'], img['file_name'])

            bbox = process_bbox(ann['bbox'], (img['height'], img['width']), cfg.HMR.input_img_shape, expand_ratio=cfg.DATASET.bbox_expand_ratio) 
            if bbox is None: continue
            
            joint_img = np.array(ann['openpose_result'], dtype=np.float32).reshape(18,-1)
            joint_img = transform_joint_to_other_db(joint_img, self.openpose_joints_name, self.joint_set['joints_name'])
            joint_img_valid = (joint_img[:,[-1]] > 0.3).astype(dtype=np.float32)
            joint_img = np.concatenate((joint_img[:,:2], joint_img_valid), axis=-1).astype(np.float32)

            joint_cam = np.array(ann['joint_cam'], dtype=np.float32).reshape(-1, 3)
            
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
            smpl_param = {k: np.array(v, dtype=np.float32) if isinstance(v, list) else v for k,v in ann['smpl_param'].items()}

            seq_name = img['sequence'] + '_' + str(ann['person_id'])
            seq_names.append(seq_name)

            datalist.append({
                'ann_id': aid,
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'joint_img': joint_img, 
                'joint_cam': joint_cam,
                'cam_param': cam_param,
                'smpl_param': smpl_param,
                'seq_name': seq_name
                })

        self.seq_names = np.unique(np.array(seq_names))
        vid_indices = split_into_chunks(np.array(seq_names), cfg.MD.seqlen, self.stride)
        return datalist, vid_indices
