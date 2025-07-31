import time
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from torch.nn import Softmax

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(x_combined)
        attention_map = self.sigmoid(attention)
        out = x * attention_map
        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class EnhancedAtt(nn.Module):
    def __init__(self, channels=8):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.res_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x_branch1, x_branch2):
        branch1_out = self.branch1(x_branch1)
        branch2_out = self.branch2(x_branch2)
        branch2_out = branch2_out.expand_as(branch1_out)
        out = self.res_conv(branch1_out * branch2_out) + branch1_out
        return out


class EnhancedMainBackbone(nn.Module):
    def __init__(self, in_channels=32, split_parts=4):
        super().__init__()
        assert in_channels % split_parts == 0
        self.split_parts = split_parts
        self.part_channels = in_channels // split_parts

        self.init_transform = nn.Sequential(
            nn.Conv2d(in_channels, self.part_channels, kernel_size=1),
            nn.BatchNorm2d(self.part_channels),
            nn.GELU()
        )

        self.conv_paths = nn.ModuleDict({
            'conv3': self._build_conv_block(self.part_channels, 3),
            'conv5': self._build_conv_block(self.part_channels, 5),
            'conv7': self._build_conv_block(self.part_channels, 7),
            'conv9': self._build_conv_block(self.part_channels, 9)
        })

        self.att_modules = nn.ModuleList([
            EnhancedAtt(self.part_channels) for _ in range(split_parts)
        ])

        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        self._init_weights()

    def _build_conv_block(self, channels, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        x_parts = torch.split(x, self.part_channels, dim=1)

        x_init = self.init_transform(x)

        conv3_out = self.conv_paths['conv3'](x_parts[0])
        att1_out = self.att_modules[0](conv3_out, x_init)

        x2_add = x_parts[1] + conv3_out
        conv5_out = self.conv_paths['conv5'](x2_add)
        att2_out = self.att_modules[1](conv5_out, conv3_out)

        x3_add = x_parts[2] + conv5_out
        conv7_out = self.conv_paths['conv7'](x3_add)
        att3_out = self.att_modules[2](conv7_out, conv5_out)

        x4_add = x_parts[3] + conv7_out
        conv9_out = self.conv_paths['conv9'](x4_add)
        att4_out = self.att_modules[3](conv9_out, conv7_out)

        out = torch.cat([att1_out, att2_out, att3_out, att4_out], dim=1)
        out = self.final_fusion(out) + residual
        return out


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def INF(B, H, W, device):
    return -torch.diag(torch.tensor(float("inf"),device=device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CC_module(nn.Module):

    def __init__(self, in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        device = x.device
        m_batchsize, _, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_H = torch.bmm(proj_query_H, proj_key_H)
        energy_H = (energy_H + INF(m_batchsize, height, width, device)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W)
        energy_W = energy_W.view(m_batchsize, height, width, width)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1))
        out_H = out_H.view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)

        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1))
        out_W = out_W.view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x

class CrissCrossAttention(nn.Module):

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.cc1 = CC_module(in_dim)
        self.cc2 = CC_module(in_dim)

    def forward(self, x):
        x = self.cc1(x)
        x = self.cc2(x)
        return x

class SSANet(nn.Module):
    def __init__(self, num_classes, n_bands, ps, inplanes, num_blocks=4, num_heads=4, num_encoders=1):
        super(SSANet, self).__init__()
        self.inplanes = inplanes
        self.input_featuremap_dim = self.inplanes
        self.featuremap_dim = self.input_featuremap_dim
        self.ps = ps
        self.sa = SpatialAttention(kernel_size=7)
        self.conv1 = nn.Conv2d(n_bands, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
        self.backbone = nn.ModuleList([EnhancedMainBackbone(in_channels=self.featuremap_dim, split_parts=4)for _ in range(num_blocks)])
        self.cca = nn.ModuleList([CrissCrossAttention(self.featuremap_dim) for _ in range(num_encoders)])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(self.featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.Spectral = Spectral(n_bands, num_classes)
    def forward(self, x):
        x = x + self.sa(x)
        x = self.conv1(x)
        x = self.bn1(x)
        for backbone in self.backbone:
            x = backbone(x)
        for encoder in self.cca:
            x = encoder(x)
        x = self.avgpool(x)
        feature_map = x
        x = x.view(x.size(0), -1)
        self.feature = x.detach()
        x = self.fc(x)
        return x, feature_map


if __name__ == "__main__":
    sSANet = SSANet(6, 64, 65, 65)
    img = torch.FloatTensor(size=(6, 64, 65, 65)).normal_(0, 1)
    output = sSANet(img)
    print(output)
