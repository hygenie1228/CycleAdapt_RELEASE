import numpy as np
import torch
import os.path as osp
import smplx

from funcs_utils import transform_joint_to_other_db

class SMPL(object):
    def __init__(self):
        self.model_path = osp.join('data', 'base_data', 'human_models')
        self.layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(self.model_path, 'smpl', gender='NEUTRAL', **self.layer_arg), 'male': smplx.create(self.model_path, 'smpl', gender='MALE', **self.layer_arg), 'female': smplx.create(self.model_path, 'smpl', gender='FEMALE', **self.layer_arg)}
        
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces
        self.pose_dim = 72
        self.shape_dim = 10

        self.orig_joint_num = 24 
        self.orig_joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
        self.orig_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.orig_skeleton = ((0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
                        (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_regressor = self.layer['neutral'].J_regressor.numpy().astype(np.float32)
        
        self.joint_num = 29
        self.joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
        self.flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
        self.skeleton = ((0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
                        (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15), (15, 24), (24, 25), (25, 27), (24, 26), (26, 28))
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.joint_regressor = self.make_joint_regressor()

        self.h36m_joint_regressor = np.load(osp.join('data', 'base_data', 'human_models', 'J_regressor_h36m_smpl.npy'))
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_Top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_root_joint_idx = 0
        self.h36m_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.h36m_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.h36m_eval_joints = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
    
    def make_joint_regressor(self):
        joint_regressor = transform_joint_to_other_db(self.orig_joint_regressor, self.orig_joints_name, self.joints_name)
        joint_regressor[self.joints_name.index('Nose')] = np.eye(self.vertex_num)[331]
        joint_regressor[self.joints_name.index('L_Eye')] = np.eye(self.vertex_num)[2802]
        joint_regressor[self.joints_name.index('R_Eye')] = np.eye(self.vertex_num)[6262]
        joint_regressor[self.joints_name.index('L_Ear')] = np.eye(self.vertex_num)[3489]
        joint_regressor[self.joints_name.index('R_Ear')] = np.eye(self.vertex_num)[3990]
        return joint_regressor

    def get_smpl_coord(self, smpl_pose, smpl_shape, gender='neutral'):
        smpl_pose, smpl_shape = torch.tensor(smpl_pose.reshape(-1, 72)), torch.tensor(smpl_shape.reshape(-1, 10))
        output = self.layer[gender](betas=smpl_shape, body_pose=smpl_pose[:, 3:], global_orient=smpl_pose[:, :3])
        smpl_mesh_cam = output.vertices
        smpl_joint_cam = torch.matmul(torch.tensor(self.joint_regressor[None,:,:]), smpl_mesh_cam)
        smpl_joint_cam = smpl_joint_cam - smpl_joint_cam[:,None,smpl.root_joint_idx,:]
        return smpl_mesh_cam.squeeze().numpy(), smpl_joint_cam.squeeze().numpy()

smpl = SMPL()