import numpy as np
import torch
import math
import cv2
from torch.nn import functional as F
from core.config import cfg
from human_models import smpl

def eval_3d_pose(pred, target):
    pred, target = pred.copy(), target.copy()
    batch_size = pred.shape[0]
    
    pred, target = pred - pred[:, None, smpl.h36m_root_joint_idx, :], target - target[:, None, smpl.h36m_root_joint_idx, :]
    pred, target = pred[:, smpl.h36m_eval_joints, :], target[:, smpl.h36m_eval_joints, :]
    
    mpjpe, pa_mpjpe = [], []
    for j in range(batch_size):
        mpjpe.append(eval_mpjpe(pred[j], target[j]))
        pa_mpjpe.append(eval_pa_mpjpe(pred[j], target[j]))
    
    return mpjpe, pa_mpjpe

def eval_mesh(pred, target, pred_joint_cam, gt_joint_cam):
    pred, target = pred.copy(), target.copy()
    batch_size = pred.shape[0]
    
    pred, target = pred - pred_joint_cam[:, None, smpl.h36m_root_joint_idx, :], target - gt_joint_cam[:, None, smpl.h36m_root_joint_idx, :]
    
    mpvpe = []
    for j in range(batch_size):
        mpvpe.append(eval_mpjpe(pred[j], target[j]))
    
    return mpvpe

def eval_accel_error(joints_pred, joints_gt, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    joints_pred, joints_gt = joints_pred.copy(), joints_gt.copy()    
    joints_pred, joints_gt = joints_pred - joints_pred[:, None, smpl.h36m_root_joint_idx, :], joints_gt - joints_gt[:, None, smpl.h36m_root_joint_idx, :]
    joints_pred, joints_gt = joints_pred[:, smpl.h36m_eval_joints, :], joints_gt[:, smpl.h36m_eval_joints, :]

    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def eval_mpjpe(predicted, target):
    return np.mean(np.sqrt(np.sum((predicted - target) ** 2, 1)))

def eval_pa_mpjpe(predicted, target):        
    predicted = rigid_align(predicted, target)
    return eval_mpjpe(predicted, target)

def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def eval_2d_joint_accuracy(predicted, target, image_size, thr=0.5):
    idx = list(range(predicted.shape[1]))
    norm = np.ones((predicted.shape[0], 2)) * np.array(image_size) / 10
    
    dists = calc_dists(predicted, target, norm)
    
    acc = np.zeros((len(idx)))
    avg_acc = 0
    cnt = 0
    
    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]], thr)
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    return acc, avg_acc
    
def eval_hmr(pred_joint_cam, gt_joint_cam, pred_mesh_cam=None, gt_mesh_cam=None, metrics=None):
    eval_dict = {}
    eval_dict['mpjpe'], eval_dict['pa-mpjpe'] = eval_3d_pose(pred_joint_cam, gt_joint_cam)

    if 'mpvpe' in metrics:
        eval_dict['mpvpe'] = eval_mesh(pred_mesh_cam, gt_mesh_cam, pred_joint_cam, gt_joint_cam)
    return eval_dict
