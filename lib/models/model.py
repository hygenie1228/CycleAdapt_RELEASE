import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy
import os.path as osp

from models.resnet import PoseResNet
from models.module import HMRNet, MDNet
from core.config import cfg
from core.logger import logger
from train_utils import load_checkpoint
from funcs_utils import rot6d_to_axis_angle, rot6d_to_rotmat, batch_rodrigues, rotmat_to_6d
from human_models import smpl

class AdaptModel(nn.Module):
    def __init__(self, hmr_net, md_net):
        super(AdaptModel, self).__init__()
        self.hmr_net = hmr_net
        self.md_net = md_net
        self.smpl_layer = copy.deepcopy(smpl.layer['neutral']).cuda()
        self.init_weights()
    
    def init_weights(self):
        self.hmr_net.init_weights()
        self.md_net.init_weights()
    
    def load_weights(self, checkpoint):
        if isinstance(checkpoint, tuple):
            self.hmr_net.load_weights(checkpoint[0])
            self.md_net.load_weights(checkpoint[1])
        else:
            self.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, inputs, mode, meta_info=None):
        if mode == 'hmr':
            return self.forward_hmr(inputs)
        elif mode == 'md':
            return self.forward_md(inputs, meta_info)
        
    def forward_hmr(self, inputs):
        x = inputs['img']

        pred_pose6d, pred_shape, pred_cam = self.hmr_net(x)
        pred_rotmat = rot6d_to_rotmat(pred_pose6d.reshape(-1,6)).reshape(-1, 24, 3, 3)
        pred_pose = rot6d_to_axis_angle(pred_pose6d.reshape(-1,6)).reshape(-1, 72)

        bs = x.shape[0]
        pred_output = self.smpl_layer(betas=pred_shape, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,[0]], pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joint_cam = torch.matmul(torch.from_numpy(smpl.joint_regressor).cuda()[None,:,:], pred_vertices)

        pred_vertices = pred_vertices + pred_cam[:,None,:]
        pred_joint_cam = pred_joint_cam + pred_cam[:,None,:]

        # project 3D coordinates to 2D space
        x = pred_joint_cam[:,:,0] / (pred_joint_cam[:,:,2] + 1e-4) * cfg.CAMERA.focal[0] + cfg.CAMERA.princpt[0]
        y = pred_joint_cam[:,:,1] / (pred_joint_cam[:,:,2] + 1e-4) * cfg.CAMERA.focal[1] + cfg.CAMERA.princpt[1]
        pred_joint_proj = torch.stack((x,y),2)       
        pred_joint_cam = pred_joint_cam - pred_joint_cam[:,smpl.root_joint_idx,None,:]

        return {
            'pose': pred_pose,
            'rotmat': pred_rotmat.reshape(bs, -1),
            'shape': pred_shape,
            'trans': pred_cam,
            'mesh_cam': pred_vertices,
            'joint_cam': pred_joint_cam,
            'joint_proj': pred_joint_proj
        }
    
    def forward_md(self, inputs, meta_info):
        x = inputs['poses']
        mask = meta_info['mask'][:,:,None]

        x = x * mask
        x = self.md_net(x)

        return {
            'poses': x,
        }

def get_model(is_train):
    hmr_net = HMRNet()
    md_net = MDNet()
    return AdaptModel(hmr_net, md_net)

