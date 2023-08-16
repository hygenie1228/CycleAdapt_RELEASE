import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

def make_linear_layers(feat_dims, relu_final=False, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, use_bn=True, bnrelu_final=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            if use_bn:
                layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class LocallyConnected2d(nn.Module):
    # Copyright©2019 Max-Planck-Gesellschaft zur Förderung
    # der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
    # for Intelligent Systems. All rights reserved.
    #
    # Contact: ps-license@tuebingen.mpg.de

    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1]), requires_grad=True
            )
        else:
            self.register_parameter('bias', None)
            
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class KeypointAttention(nn.Module):
    def __init__(self, use_conv=False, in_channels=(256, 64), out_channels=(256, 64), act='softmax', use_scale=False):
        super(KeypointAttention, self).__init__()
        self.use_conv = use_conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.use_scale = use_scale
        if use_conv:
            self.conv1x1_pose = nn.Conv1d(in_channels[0], out_channels[0], kernel_size=1)
            self.conv1x1_shape_cam = nn.Conv1d(in_channels[1], out_channels[1], kernel_size=1)

    def forward(self, features, heatmaps):
        batch_size, num_joints, height, width = heatmaps.shape

        if self.use_scale:
            scale = 1.0 / np.sqrt(height * width)
            heatmaps = heatmaps * scale

        if self.act == 'softmax':
            normalized_heatmap = F.softmax(heatmaps.reshape(batch_size, num_joints, -1), dim=-1)
        elif self.act == 'sigmoid':
            normalized_heatmap = torch.sigmoid(heatmaps.reshape(batch_size, num_joints, -1))
        features = features.reshape(batch_size, -1, height*width)
        
        attended_features = torch.matmul(normalized_heatmap, features.transpose(2,1))
        attended_features = attended_features.transpose(2,1)

        if self.use_conv:
            if attended_features.shape[1] == self.in_channels[0]:
                attended_features = self.conv1x1_pose(attended_features)
            else:
                attended_features = self.conv1x1_shape_cam(attended_features)

        return attended_features