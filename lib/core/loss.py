import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.config import cfg
from core.prior import MaxMixturePrior

from human_models import smpl

class CLSLoss(nn.Module):
    def __init__(self):
        super(CLSLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        pred, target = pred.reshape(-1), target.reshape(-1)
        valid = (target != 0)

        pred, target = pred[valid], target[valid]
        target[target!=1] = 0

        loss = self.bce_loss(pred, target.float())
        return loss.mean()

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, valid=None):
        target = target.to(pred.device)

        if valid is not None:
            valid = valid.to(pred.device)
            pred, target = pred * valid[...,None], target * valid[...,None]

        loss = self.criterion(pred, target)
        return loss
    
class ParamLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ParamLoss, self).__init__()        
        if type == 'l1':
            self.criterion = nn.L1Loss(reduction='mean')
        elif type == 'l2':
            self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, param_out, param_gt, valid=None):
        param_out = param_out.reshape(-1, param_out.shape[-1])
        param_gt = param_gt.reshape(-1, param_gt.shape[-1]).to(param_out.device)

        if valid is not None:
            valid = valid.reshape(-1).to(param_out.device)
            param_out, param_gt = param_out * valid[:,None], param_gt * valid[:,None]
        
        loss = self.criterion(param_out, param_gt)
        return loss

class AccelLoss(nn.Module):
    def __init__(self, type='l1'):
        super(AccelLoss, self).__init__()        
        if type == 'l1':
            self.criterion = nn.L1Loss(reduction='mean')
        elif type == 'l2':
            self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, inp, pred, accel_gt, valid=None):
        accel_gt = accel_gt.to(pred.device)

        accel_pred = pred[:,:-2,:] - 2 * pred[:,1:-1,:] + pred[:,2:,:]
        loss = self.criterion(accel_pred, accel_gt)
        return loss

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, param, valid=None):
        valid = valid.bool().all(1)
        param = param * valid[:,None,None].to(param.device)

        zeros = torch.zeros_like(param, device='cuda')
        loss = self.criterion(param, zeros)
        return loss.mean()

class HeatmapMSELoss(nn.Module):
    def __init__(self, has_valid):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.has_valid = has_valid

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        target_weight = target_weight.reshape((batch_size, num_joints, 1))
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.has_valid:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class PriorLoss(nn.Module):
    def __init__(self):
        super(PriorLoss, self).__init__()
        self.pose_prior = MaxMixturePrior(prior_folder='data/base_data/pose_prior', num_gaussians=8, dtype=torch.float32).cuda()

        self.pose_prior_weight = 1.0
        self.shape_prior_weight = 0.2
        self.angle_prior_weight = 0.0
        
    def forward(self, body_pose, betas=None):
        # Pose prior loss
        pose_prior_loss = self.pose_prior_weight * self.pose_prior(body_pose, betas)
        # Angle prior for knees and elbows
        angle_prior_loss = self.angle_prior_weight * self.angle_prior(body_pose).sum(dim=-1)
        # Regularizer to prevent betas from taking large values
        shape_prior_loss = self.shape_prior_weight * (betas ** 2).sum(dim=-1)
        loss = pose_prior_loss + angle_prior_loss + shape_prior_loss
        return loss.mean()
    
    def angle_prior(self, pose):
        """
        Angle prior that penalizes unnatural bending of the knees and elbows
        """
        # We subtract 3 because pose does not include the global rotation of the model
        return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2
    

def get_loss():
    loss = {}
    loss['param'] = ParamLoss(type='l1')
    loss['proj'] = CoordLoss()
    loss['joint'] = CoordLoss()
    loss['prior'] = PriorLoss()
    loss['reg'] = RegLoss()    
    return loss
