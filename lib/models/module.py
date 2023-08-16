import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os.path as osp
import math
import torchvision.models.resnet as resnet

from core.config import cfg
from models.layer import make_linear_layers, make_conv_layers, make_deconv_layers, LocallyConnected2d, KeypointAttention

def gn_helper(planes):
    if 0:
        return nn.BatchNorm2d(planes)
    else:
        return nn.GroupNorm(32 // 8, planes)


class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer=gn_helper, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HMRNet(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, gn=True):
        block = Bottleneck
        layers = [3, 4, 6, 3]
        norm_layer = gn_helper

        self.inplanes = 64
        super(HMRNet, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(norm_layer, block, 64, layers[0])
        self.layer2 = self._make_layer(norm_layer, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(norm_layer, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(norm_layer, block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        smpl_mean_params = osp.join('data', 'base_data', 'human_models', 'smpl_mean_params.npz')
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, norm_layer, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, norm_layer, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer))

        return nn.Sequential(*layers)

    def init_weights(self):
        resnet_imagenet = resnet.resnet50(pretrained=True)
        self.load_state_dict(resnet_imagenet.state_dict(),strict=False)

    def load_weights(self, checkpoint):
        from train_utils import check_data_parallel
        self.load_state_dict(check_data_parallel(checkpoint['model']), strict=False)        

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_cam = torch.stack([pred_cam[:,1],
                                pred_cam[:,2],
                                2*cfg.CAMERA.focal[0]/(cfg.HMR.input_img_shape[0]*pred_cam[:,0] +1e-9)],dim=-1)

        return pred_pose, pred_shape, pred_cam


class MDNet(nn.Module):
    def __init__(self):
        super(MDNet, self).__init__()
        self.motion_fc_in = nn.Linear(cfg.MD.hidden_dim, cfg.MD.hidden_dim)
        self.motion_mlp = TransMLP(cfg.MD.hidden_dim, cfg.MD.seqlen, cfg.MD.num_layers)
        self.motion_fc_out = nn.Linear(cfg.MD.hidden_dim, cfg.MD.hidden_dim)
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def load_weights(self, checkpoint):
        from train_utils import check_data_parallel
        self.load_state_dict(check_data_parallel(checkpoint['model']), strict=False)  

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.motion_fc_in(x)
        x = x.permute(0, 2, 1)
        x = self.motion_mlp(x)
        x = x.permute(0, 2, 1)
        x = self.motion_fc_out(x)
        return x
    


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y



class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPblock(nn.Module):
    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False):
        super().__init__()
        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)

        layernorm_axis = 'spatial'
        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_
        return x


class TransMLP(nn.Module):
    def __init__(self, dim, seq, num_layers):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPblock(dim, seq)
            for i in range(num_layers)])

    def forward(self, x):
        x = self.mlps(x)
        return x