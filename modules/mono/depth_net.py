"""
This is module predict the structural ray scan from perspective image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import *

import os
from pathlib import Path
import numpy as np
import torch

from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

from modules.network_utils import *


class depth_net(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_res()

    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (
            1 / self.d_hyp
        )  # (D,)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # weighted average
        d = torch.sum(prob * d_vals, dim=-1)  # (N, fW)

        return d, attn, prob


class depth_feature_res(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Resnet backbone:
        (480, 640, 3) -> Resnet50(3) -> (30, 40, 1024) # this is the conv feature

        Attn:
        img (480, 640, 3) => ResNet (30, 40, 32) => Avg_Pool (1, 40, 32) ==> Q_proj (1, 40, 32) ==> Attention (1, 40, 32) # this is the attn feature
                            (+ positional encoding)  (+ positional encoding)                             / |
                                            | \=============================> K_proj (30, 40, 32) ======/  |
                                            |===============================> V_proj (30, 40, 32) =========|

        """
        res50 = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, False, True]
        )
        self.resnet = nn.Sequential(
            IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        )
        self.conv = ConvBnReLU(
            in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        )

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        # Attention block, attention to 2D image
        self.q_proj = nn.Linear(160, 128, bias=False) # 40x40=160
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        x = self.resnet(x)["feat"]  # (N, 1024, fH, fW)
        x = self.conv(x)  # (N, 32, fH, fW)
        fH, fW = list(x.shape[2:])
        N = x.shape[0]

        # reduce vertically
        query = x.mean(dim=2)  # (N, 32, fW)

        # channel last
        query = query.permute(0, 2, 1)  # (N, fW, 32)

        # reshape from (N, 32, fH, fW) to (N, 32, fHxfW)
        x = x.view(list(x.shape[:2]) + [-1])  # (N, 32, fHxfW)
        # channel last to cope with fc
        x = x.permute(0, 2, 1)  # (N, fHxfW, 32)

        # compute 2d positional encoding here
        # todo:
        # Example: for (4, 4) image
        # | (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), (-1.5, 1.5)|
        # | (-0.5, -1.5), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5)|
        # | (0.5, -1.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 1.5)|
        # | (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)|
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)  # (fH, fW, 2)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (fH, fW, 32)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))  # (1, fHxfW, 32)
        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
        x = torch.cat((x, pos_enc_2d), dim=-1)  # (N, fHxfW, 32+32)

        # get the 1d positional encoding here
        # todo:
        # Example: for (5, ) ray
        # |-2, -1, 0, 1, 2|
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5  # (fW,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))  # (fW, 32)
        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))  # (N, fW, 32)
        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, fW, 32+32)

        # attention
        query = self.q_proj(query)  # (N, fW, 32)
        key = self.k_proj(x)  # (N, fHxfW, 32)
        value = self.v_proj(x)  # (N, fHxfW, 32)

        # resize the mask
        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
                torch.bool
            )  # (N, fH, fW)
            mask = torch.logical_not(
                mask
            )  # True is not allow to attend, original mask as True on valid values
            mask = mask.reshape((mask.shape[0], 1, -1))  # (N, 1, fHxfW)
            # same mask for all fW
            mask = mask.repeat(1, fW, 1)  # (N, fW, fHxfW)
        x, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, fW, 32)

        return x, attn_w



class depth_net_metric3d(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_metric3d()
        
    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (
            1 / self.d_hyp
        )  # (D,)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # weighted average
        d = torch.sum(prob * d_vals, dim=-1)  # (N, fW)

        return d, attn, prob


class depth_feature_metric3d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Resnet backbone:
        (480, 640, 3) -> Resnet50(3) -> (30, 40, 1024) # this is the conv feature

        Attn:
        img (480, 640, 3) => ResNet (30, 40, 32) => Avg_Pool (1, 40, 32) ==> Q_proj (1, 40, 32) ==> Attention (1, 40, 32) # this is the attn feature
                            (+ positional encoding)  (+ positional encoding)                             / |
                                            | \=============================> K_proj (30, 40, 32) ======/  |
                                            |===============================> V_proj (30, 40, 32) =========|

        """
        # res50 = resnet50(
        #     pretrained=True, replace_stride_with_dilation=[False, False, True]
        # )
        # self.resnet = nn.Sequential(
        #     IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        # )
        # self.conv = ConvBnReLU(
        #     in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        # )

        ##### INSERTED HERE:

        self.cnn = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output shape: (16, H/2, W/2)

            # Conv Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output shape: (32, H/4, W/4)

            # Conv Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output shape: (64, H/8, W/8)

            # Conv Block 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Output shape: (128, H/16, W/16)
            )

        #####

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        # Attention block, attention to 2D image
        self.q_proj = nn.Linear(160, 128, bias=False) # 128 + 32 = 160
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        # x = self.resnet(x)["feat"]  # (N, 1024, fH, fW)
        # x = self.conv(x)  # (N, 32, fH, fW)

        ##### INSERT HERE:
        if x.dim() == 3:  # [batch_size, height, width]
            x = x.unsqueeze(1)  # Now x is [batch_size, 1, height, width]
        #print("x.shape: ", x.shape)
        x = self.cnn(x)  # (N, 128, H/16, W/16)
        #####

        fH, fW = list(x.shape[2:])
        N = x.shape[0]

        # reduce vertically
        query = x.mean(dim=2)  # (N, 128, fW)

        # channel last
        query = query.permute(0, 2, 1)  # (N, fW, 128)

        # reshape from (N, 128, fH, fW) to (N, 128, fHxfW)
        x = x.view(list(x.shape[:2]) + [-1])  # (N, 128, fHxfW)
        # channel last to cope with fc
        x = x.permute(0, 2, 1)  # (N, fHxfW, 128)

        # compute 2d positional encoding here
        # todo:
        # Example: for (4, 4) image
        # | (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), (-1.5, 1.5)|
        # | (-0.5, -1.5), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5)|
        # | (0.5, -1.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 1.5)|
        # | (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)|
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)  # (fH, fW, 2)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (fH, fW, 32)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))  # (1, fHxfW, 32)
        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
        x = torch.cat((x, pos_enc_2d), dim=-1)  # (N, fHxfW, 160)

        # get the 1d positional encoding here
        # todo:
        # Example: for (5, ) ray
        # |-2, -1, 0, 1, 2|
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5  # (fW,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))  # (fW, 32)
        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))  # (N, fW, 32)
        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, fW, 160)

        # attention
        query = self.q_proj(query)  # (N, fW, 128)
        key = self.k_proj(x)  # (N, fHxfW, 128)
        value = self.v_proj(x)  # (N, fHxfW, 128)

        # resize the mask
        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
                torch.bool
            )  # (N, fH, fW)
            mask = torch.logical_not(
                mask
            )  # True is not allow to attend, original mask as True on valid values
            mask = mask.reshape((mask.shape[0], 1, -1))  # (N, 1, fHxfW)
            # same mask for all fW
            mask = mask.repeat(1, fW, 1)  # (N, fW, fHxfW)
        x, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, fW, 128)

        return x, attn_w




class depth_net_metric3d_uncertainty(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_metric3d()

        # New fully connected layers to predict location and scale
        self.fc_loc = nn.Linear(128, 1)  # Location parameter (mu)
        self.fc_scale = nn.Linear(128, 1)  # Scale parameter (b)


    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        #d_vals = torch.linspace(
        #    self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        #) ** (
        #    1 / self.d_hyp
        #)  # (D,)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # Predict location (mu) and scale (b) for each fW interval
        # For each width interval, you get one location and one scale
        loc = self.fc_loc(x).squeeze(-1)  # (N, fW), location for each interval
        scale = F.softplus(self.fc_scale(x)).squeeze(-1)  # (N, fW), ensure scale is positive

        return loc, scale, attn, prob
    


class depth_net_metric3d_depths_normals(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_metric3d_depths_normals()
        
    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (
            1 / self.d_hyp
        )  # (D,)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # weighted average
        d = torch.sum(prob * d_vals, dim=-1)  # (N, fW)

        return d, attn, prob


class depth_feature_metric3d_depths_normals(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Resnet backbone:
        (480, 640, 3) -> Resnet50(3) -> (30, 40, 1024) # this is the conv feature

        Attn:
        img (480, 640, 3) => ResNet (30, 40, 32) => Avg_Pool (1, 40, 32) ==> Q_proj (1, 40, 32) ==> Attention (1, 40, 32) # this is the attn feature
                            (+ positional encoding)  (+ positional encoding)                             / |
                                            | \=============================> K_proj (30, 40, 32) ======/  |
                                            |===============================> V_proj (30, 40, 32) =========|

        """
        # res50 = resnet50(
        #     pretrained=True, replace_stride_with_dilation=[False, False, True]
        # )
        # self.resnet = nn.Sequential(
        #     IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        # )
        # self.conv = ConvBnReLU(
        #     in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        # )

        ##### INSERT HERE:

        self.cnn = nn.Sequential(
            # Conv Block 1
            #nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output shape: (16, H/2, W/2)

            # Conv Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output shape: (32, H/4, W/4)

            # Conv Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output shape: (64, H/8, W/8)

            # Conv Block 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Output shape: (128, H/16, W/16)
            )

        #####

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        # Attention block, attention to 2D image
        self.q_proj = nn.Linear(160, 128, bias=False) # 128 + 32 = 160
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        # x = self.resnet(x)["feat"]  # (N, 1024, fH, fW)
        # x = self.conv(x)  # (N, 32, fH, fW)

        ##### INSERT HERE:
        if x.dim() == 3:  # [batch_size, height, width]
            x = x.unsqueeze(1)  # Now x is [batch_size, 1, height, width]
        #print("x.shape: ", x.shape)
        x = self.cnn(x)  # (N, 128, H/16, W/16)
        #####

        fH, fW = list(x.shape[2:])
        N = x.shape[0]

        # reduce vertically
        query = x.mean(dim=2)  # (N, 128, fW)

        # channel last
        query = query.permute(0, 2, 1)  # (N, fW, 128)

        # reshape from (N, 128, fH, fW) to (N, 128, fHxfW)
        x = x.view(list(x.shape[:2]) + [-1])  # (N, 128, fHxfW)
        # channel last to cope with fc
        x = x.permute(0, 2, 1)  # (N, fHxfW, 128)

        # compute 2d positional encoding here
        # todo:
        # Example: for (4, 4) image
        # | (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), (-1.5, 1.5)|
        # | (-0.5, -1.5), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5)|
        # | (0.5, -1.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 1.5)|
        # | (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)|
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)  # (fH, fW, 2)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (fH, fW, 32)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))  # (1, fHxfW, 32)
        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
        x = torch.cat((x, pos_enc_2d), dim=-1)  # (N, fHxfW, 160)

        # get the 1d positional encoding here
        # todo:
        # Example: for (5, ) ray
        # |-2, -1, 0, 1, 2|
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5  # (fW,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))  # (fW, 32)
        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))  # (N, fW, 32)
        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, fW, 160)

        # attention
        query = self.q_proj(query)  # (N, fW, 128)
        key = self.k_proj(x)  # (N, fHxfW, 128)
        value = self.v_proj(x)  # (N, fHxfW, 128)

        # resize the mask
        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
                torch.bool
            )  # (N, fH, fW)
            mask = torch.logical_not(
                mask
            )  # True is not allow to attend, original mask as True on valid values
            mask = mask.reshape((mask.shape[0], 1, -1))  # (N, 1, fHxfW)
            # same mask for all fW
            mask = mask.repeat(1, fW, 1)  # (N, fW, fHxfW)
        x, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, fW, 128)

        return x, attn_w




class depth_net_metric3d_depths_normals_segmentation(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_metric3d_depths_normals_segmentation()
        
    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (
            1 / self.d_hyp
        )  # (D,)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # weighted average
        d = torch.sum(prob * d_vals, dim=-1)  # (N, fW)

        return d, attn, prob


class depth_feature_metric3d_depths_normals_segmentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Resnet backbone:
        (480, 640, 3) -> Resnet50(3) -> (30, 40, 1024) # this is the conv feature

        Attn:
        img (480, 640, 3) => ResNet (30, 40, 32) => Avg_Pool (1, 40, 32) ==> Q_proj (1, 40, 32) ==> Attention (1, 40, 32) # this is the attn feature
                            (+ positional encoding)  (+ positional encoding)                             / |
                                            | \=============================> K_proj (30, 40, 32) ======/  |
                                            |===============================> V_proj (30, 40, 32) =========|

        """
        # res50 = resnet50(
        #     pretrained=True, replace_stride_with_dilation=[False, False, True]
        # )
        # self.resnet = nn.Sequential(
        #     IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        # )
        # self.conv = ConvBnReLU(
        #     in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        # )

        ##### INSERT HERE:

        self.cnn = nn.Sequential(
            # Conv Block 1
            #nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=4+150, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output shape: (16, H/2, W/2)

            # Conv Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output shape: (32, H/4, W/4)

            # Conv Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output shape: (64, H/8, W/8)

            # Conv Block 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Output shape: (128, H/16, W/16)
            )

        #####

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        # Attention block, attention to 2D image
        self.q_proj = nn.Linear(160, 128, bias=False) # 128 + 32 = 160
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        # x = self.resnet(x)["feat"]  # (N, 1024, fH, fW)
        # x = self.conv(x)  # (N, 32, fH, fW)

        ##### INSERT HERE:
        if x.dim() == 3:  # [batch_size, height, width]
            x = x.unsqueeze(1)  # Now x is [batch_size, 1, height, width]
        #print("x.shape: ", x.shape)
        x = self.cnn(x)  # (N, 128, H/16, W/16)
        #####

        fH, fW = list(x.shape[2:])
        N = x.shape[0]

        # reduce vertically
        query = x.mean(dim=2)  # (N, 128, fW)

        # channel last
        query = query.permute(0, 2, 1)  # (N, fW, 128)

        # reshape from (N, 128, fH, fW) to (N, 128, fHxfW)
        x = x.view(list(x.shape[:2]) + [-1])  # (N, 128, fHxfW)
        # channel last to cope with fc
        x = x.permute(0, 2, 1)  # (N, fHxfW, 128)

        # compute 2d positional encoding here
        # todo:
        # Example: for (4, 4) image
        # | (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), (-1.5, 1.5)|
        # | (-0.5, -1.5), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5)|
        # | (0.5, -1.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 1.5)|
        # | (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)|
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)  # (fH, fW, 2)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (fH, fW, 32)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))  # (1, fHxfW, 32)
        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
        x = torch.cat((x, pos_enc_2d), dim=-1)  # (N, fHxfW, 160)

        # get the 1d positional encoding here
        # todo:
        # Example: for (5, ) ray
        # |-2, -1, 0, 1, 2|
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5  # (fW,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))  # (fW, 32)
        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))  # (N, fW, 32)
        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, fW, 160)

        # attention
        query = self.q_proj(query)  # (N, fW, 128)
        key = self.k_proj(x)  # (N, fHxfW, 128)
        value = self.v_proj(x)  # (N, fHxfW, 128)

        # resize the mask
        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
                torch.bool
            )  # (N, fH, fW)
            mask = torch.logical_not(
                mask
            )  # True is not allow to attend, original mask as True on valid values
            mask = mask.reshape((mask.shape[0], 1, -1))  # (N, 1, fHxfW)
            # same mask for all fW
            mask = mask.repeat(1, fW, 1)  # (N, fW, fHxfW)
        x, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, fW, 128)

        return x, attn_w




class depth_net_uncertainty(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_res_uncertainty()

        # New fully connected layers to predict location and scale
        self.fc_loc = nn.Linear(128, 1)  # Location parameter (mu)
        self.fc_scale = nn.Linear(128, 1)  # Scale parameter (b)


    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (
            1 / self.d_hyp
        )  # (D,)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # Predict location (mu) and scale (b) for each fW interval
        # For each width interval, you get one location and one scale
        loc = self.fc_loc(x).squeeze(-1)  # (N, fW), location for each interval
        scale = F.softplus(self.fc_scale(x)).squeeze(-1)  # (N, fW), ensure scale is positive

        return loc, scale, attn, prob


class depth_feature_res_uncertainty(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Resnet backbone:
        (480, 640, 3) -> Resnet50(3) -> (30, 40, 1024) # this is the conv feature

        Attn:
        img (480, 640, 3) => ResNet (30, 40, 32) => Avg_Pool (1, 40, 32) ==> Q_proj (1, 40, 32) ==> Attention (1, 40, 32) # this is the attn feature
                            (+ positional encoding)  (+ positional encoding)                             / |
                                            | \=============================> K_proj (30, 40, 32) ======/  |
                                            |===============================> V_proj (30, 40, 32) =========|

        """
        res50 = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, False, True]
        )
        self.resnet = nn.Sequential(
            IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        )
        self.conv = ConvBnReLU(
            in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        )

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        # Attention block, attention to 2D image
        self.q_proj = nn.Linear(160, 128, bias=False) # 40x40=160
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        x = self.resnet(x)["feat"]  # (N, 1024, fH, fW)
        x = self.conv(x)  # (N, 32, fH, fW)
        fH, fW = list(x.shape[2:])
        N = x.shape[0]

        # reduce vertically
        query = x.mean(dim=2)  # (N, 32, fW)

        # channel last
        query = query.permute(0, 2, 1)  # (N, fW, 32)

        # reshape from (N, 32, fH, fW) to (N, 32, fHxfW)
        x = x.view(list(x.shape[:2]) + [-1])  # (N, 32, fHxfW)
        # channel last to cope with fc
        x = x.permute(0, 2, 1)  # (N, fHxfW, 32)

        # compute 2d positional encoding here
        # todo:
        # Example: for (4, 4) image
        # | (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), (-1.5, 1.5)|
        # | (-0.5, -1.5), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5)|
        # | (0.5, -1.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 1.5)|
        # | (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)|
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)  # (fH, fW, 2)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (fH, fW, 32)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))  # (1, fHxfW, 32)
        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
        x = torch.cat((x, pos_enc_2d), dim=-1)  # (N, fHxfW, 32+32)

        # get the 1d positional encoding here
        # todo:
        # Example: for (5, ) ray
        # |-2, -1, 0, 1, 2|
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5  # (fW,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))  # (fW, 32)
        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))  # (N, fW, 32)
        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, fW, 32+32)

        # attention
        query = self.q_proj(query)  # (N, fW, 32)
        key = self.k_proj(x)  # (N, fHxfW, 32)
        value = self.v_proj(x)  # (N, fHxfW, 32)

        # resize the mask
        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
                torch.bool
            )  # (N, fH, fW)
            mask = torch.logical_not(
                mask
            )  # True is not allow to attend, original mask as True on valid values
            mask = mask.reshape((mask.shape[0], 1, -1))  # (N, 1, fHxfW)
            # same mask for all fW
            mask = mask.repeat(1, fW, 1)  # (N, fW, fHxfW)
        x, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, fW, 32)

        return x, attn_w




class depth_net_depthanything(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_depthanything()

    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (
            1 / self.d_hyp
        )  # (D,)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # weighted average
        d = torch.sum(prob * d_vals, dim=-1)  # (N, fW)

        return d, attn, prob




class depth_net_depthanything_uncertainty(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_depthanything()

        # New fully connected layers to predict location and scale
        self.fc_loc = nn.Linear(128, 1)  # Location parameter (mu)
        self.fc_scale = nn.Linear(128, 1)  # Scale parameter (b)

    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # Predict location (mu) and scale (b) for each fW interval
        # For each width interval, you get one location and one scale
        loc = self.fc_loc(x).squeeze(-1)  # (N, fW), location for each interval
        scale = F.softplus(self.fc_scale(x)).squeeze(-1)  # (N, fW), ensure scale is positive

        return loc, scale, attn, prob




class depth_net_depthanything_uncertainty_sem(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_depthanything()
        self.semantic_features = depth_feature_depthanything()

        # New fully connected layers to predict location and scale for depth
        self.fc_loc = nn.Linear(128, 1)  # Location parameter (mu)
        self.fc_scale = nn.Linear(128, 1)  # Scale parameter (b)

        # New fully connected layer for semantic class prediction (3 classes)
        self.fc_class = nn.Linear(128, 3)  # Output 3 logits for the 3 classes


    def forward(self, x, mask=None):


        ### depth

        # extract depth features
        x_depth, attn = self.depth_feature(x, mask)  # (N, fW, D)

        # for probability volume using soft-max
        prob = F.softmax(x_depth, dim=-1)  # (N, fW, D)

        # Predict location (mu) and scale (b) for each fW interval
        # For each width interval, you get one location and one scale
        loc = self.fc_loc(x_depth).squeeze(-1)  # (N, fW), location for each interval
        scale = F.softplus(self.fc_scale(x_depth)).squeeze(-1)  # (N, fW), ensure scale is positive



        ### semantics

        # extract semantic features
        x_sem, attn_sem = self.semantic_features(x, mask)  # (N, fW, D)

        # for probability volume using soft-max
        prob_sem = F.softmax(x_sem, dim=-1)

        # Predict class logits
        # For each width interval, you get one set of logits for the 3 classes
        class_logits = self.fc_class(x_sem)  # (N, fW, 3), raw class scores
        class_probs = F.softmax(class_logits, dim=-1)  # (N, fW, 3), class probabilities


        return loc, scale, attn, prob, class_logits, class_probs



class depth_net_depthanything_uncertainty_sem(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_depthanything()
        self.semantic_features = depth_feature_depthanything()

        # New fully connected layers to predict location and scale for depth
        self.fc_loc = nn.Linear(128, 1)  # Location parameter (mu)
        self.fc_scale = nn.Linear(128, 1)  # Scale parameter (b)

        # New fully connected layer for semantic class prediction (3 classes)
        self.fc_class = nn.Linear(128, 3)  # Output 3 logits for the 3 classes


    def forward(self, x, mask=None):


        ### depth

        # extract depth features
        x_depth, attn = self.depth_feature(x, mask)  # (N, fW, D)

        # for probability volume using soft-max
        prob = F.softmax(x_depth, dim=-1)  # (N, fW, D)

        # Predict location (mu) and scale (b) for each fW interval
        # For each width interval, you get one location and one scale
        loc = self.fc_loc(x_depth).squeeze(-1)  # (N, fW), location for each interval
        scale = F.softplus(self.fc_scale(x_depth)).squeeze(-1)  # (N, fW), ensure scale is positive



        ### semantics

        # extract semantic features
        x_sem, attn_sem = self.semantic_features(x, mask)  # (N, fW, D)

        # for probability volume using soft-max
        prob_sem = F.softmax(x_sem, dim=-1)

        # Predict class logits
        # For each width interval, you get one set of logits for the 3 classes
        class_logits = self.fc_class(x_sem)  # (N, fW, 3), raw class scores
        class_probs = F.softmax(class_logits, dim=-1)  # (N, fW, 3), class probabilities


        return loc, scale, attn, prob, class_logits, class_probs





class depth_net_depthanything_uncertainty_sem_v2(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_depthanything()

        # New fully connected layers to predict location and scale for depth
        self.fc_loc = nn.Linear(125, 1)  # Location parameter (mu)
        self.fc_scale = nn.Linear(125, 1)  # Scale parameter (b)

        # New fully connected layer for semantic class prediction (3 classes)
        self.fc_class = nn.Linear(3, 3)  # Output 3 logits for the 3 classes


    def forward(self, x, mask=None):


        ### depth

        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)
        x_depth = x[:, :, :125]  # (N, fW, 125)
        x_sem = x[:, :, 125:]  # (N, fW, 3)

        # for probability volume using soft-max
        prob = F.softmax(x_depth, dim=-1)  # (N, fW, D)

        # Predict location (mu) and scale (b) for each fW interval
        # For each width interval, you get one location and one scale
        loc = self.fc_loc(x_depth).squeeze(-1)  # (N, fW), location for each interval
        scale = F.softplus(self.fc_scale(x_depth)).squeeze(-1)  # (N, fW), ensure scale is positive



        ### semantics

        # for probability volume using soft-max
        prob_sem = F.softmax(x_sem, dim=-1)

        # Predict class logits
        # For each width interval, you get one set of logits for the 3 classes
        class_logits = self.fc_class(x_sem)  # (N, fW, 3), raw class scores
        class_probs = F.softmax(class_logits, dim=-1)  # (N, fW, 3), class probabilities


        return loc, scale, attn, prob, class_logits, class_probs




class semantic_net_depthanything(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.semantic_features = depth_feature_depthanything()

        # New fully connected layer for semantic class prediction (3 classes)
        self.fc_class = nn.Linear(128, 3)  # Output 3 logits for the 3 classes


    def forward(self, x, mask=None):

        ### semantics
        # extract semantic features
        x_sem, attn_sem = self.semantic_features(x, mask)  # (N, fW, D)
        assert torch.isfinite(x_sem).all(), "Invalid values in semantic features"
        # Predict class logits
        class_logits = self.fc_class(x_sem)  # (N, fW, 3), raw class scores
        #class_logits = torch.clamp(class_logits, min=-1e9, max=1e9)  # Clamp logits to avoid extreme values
        class_probs = F.softmax(class_logits, dim=-1)  # (N, fW, 3), class probabilities

        return class_logits, class_probs





class depth_feature_depthanything(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Resnet backbone:
        (480, 640, 3) -> Resnet50(3) -> (30, 40, 1024) # this is the conv feature

        Attn:
        img (480, 640, 3) => ResNet (30, 40, 32) => Avg_Pool (1, 40, 32) ==> Q_proj (1, 40, 32) ==> Attention (1, 40, 32) # this is the attn feature
                            (+ positional encoding)  (+ positional encoding)                             / |
                                            | \=============================> K_proj (30, 40, 32) ======/  |
                                            |===============================> V_proj (30, 40, 32) =========|

        """
        # res50 = resnet50(
        #     pretrained=True, replace_stride_with_dilation=[False, False, True]
        # )
        # self.resnet = nn.Sequential(
        #     IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        # )

        ####

        # # Encoder
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {device}")
        # 
        # model_configs = {
        #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        # }
        # encoder =  'vitl' # or 'vitl', 'vitb', 'vits'
        # dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        # self.da = DepthAnythingV2(**model_configs[encoder])
        # self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
        # self.da = self.da.to(device)  # Move the model to GPU
        # self.da.eval()
        
        # Conv Layer
        
#        # vitl
#        self.conv = ConvBnReLU(
#            in_channels=1024, out_channels=128, kernel_size=3, padding=1, stride=1
#        )

#        # vitb
#        self.conv = ConvBnReLU(
#            in_channels=768, out_channels=128, kernel_size=3, padding=1, stride=1
#        )

#        # vits
#        self.conv = ConvBnReLU(
#            in_channels=384, out_channels=128, kernel_size=3, padding=1, stride=1
#        )

        ####

        # depthpro
        self.conv = ConvBnReLU(
            in_channels=3072, out_channels=128, kernel_size=3, padding=1, stride=1
        )

        ####        

#        ####
#        # DinoV2
#        #in_channels = 384 # vits14
#        #in_channels = 768 # vitb14
#        in_channels = 1024 # vitl14
#        self.conv = ConvBnReLU(
#            in_channels=in_channels, out_channels=128, kernel_size=3, padding=1, stride=1
#        )
#        ####

        #self.conv = ConvBnReLU(
        #    in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        #)
        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        # Attention block, attention to 2D image
        self.q_proj = nn.Linear(160, 128, bias=False) # 40x40=160
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        #self.attn = Attention() #_Assertions()
        self.attn = Attention_Assertions()

#    def forward(self, x, mask=None):
#        #x = self.resnet(x)["feat"]  # (N, 1024, fH, fW)
#        
#        # # encoder
#        # _, _, h, w = x.shape
#        # input_tensor, _ =  self.da.image2tensor_simplified(x)
#        # with torch.no_grad():
#        #     x = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
#        # x = F.interpolate(x, size=(h//16, w//16), mode='bilinear', align_corners=True) # (N, 1024, fH, fW)
#        
#        # conv layer
#        x = self.conv(x)  # (N, 32, fH, fW)
#        fH, fW = list(x.shape[2:])
#        N = x.shape[0]
#
#        # reduce vertically
#        query = x.mean(dim=2)  # (N, 32, fW)
#
#        # channel last
#        query = query.permute(0, 2, 1)  # (N, fW, 32)
#
#        # reshape from (N, 32, fH, fW) to (N, 32, fHxfW)
#        x = x.view(list(x.shape[:2]) + [-1])  # (N, 32, fHxfW)
#        # channel last to cope with fc
#        x = x.permute(0, 2, 1)  # (N, fHxfW, 32)
#
#        # compute 2d positional encoding here
#        # todo:
#        # Example: for (4, 4) image
#        # | (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), (-1.5, 1.5)|
#        # | (-0.5, -1.5), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5)|
#        # | (0.5, -1.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 1.5)|
#        # | (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)|
#        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
#        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
#        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
#        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)  # (fH, fW, 2)
#        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (fH, fW, 32)
#        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))  # (1, fHxfW, 32)
#        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
#        x = torch.cat((x, pos_enc_2d), dim=-1)  # (N, fHxfW, 32+32)
#
#        # get the 1d positional encoding here
#        # todo:
#        # Example: for (5, ) ray
#        # |-2, -1, 0, 1, 2|
#        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5  # (fW,)
#        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))  # (fW, 32)
#        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))  # (N, fW, 32)
#        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, fW, 32+32)
#
#        # attention
#        query = self.q_proj(query)  # (N, fW, 32)
#        key = self.k_proj(x)  # (N, fHxfW, 32)
#        value = self.v_proj(x)  # (N, fHxfW, 32)
#
#        # resize the mask
#        if mask is not None:
#            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
#                torch.bool
#            )  # (N, fH, fW)
#            mask = torch.logical_not(
#                mask
#            )  # True is not allow to attend, original mask as True on valid values
#            mask = mask.reshape((mask.shape[0], 1, -1))  # (N, 1, fHxfW)
#            # same mask for all fW
#            mask = mask.repeat(1, fW, 1)  # (N, fW, fHxfW)
#        x, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, fW, 32)
#
#        return x, attn_w

    def forward(self, x, mask=None):
        # conv layer
        x = self.conv(x)  # (N, 32, fH, fW)
        assert torch.isfinite(x).all(), "Invalid values after conv layer"
        
        fH, fW = list(x.shape[2:])
        N = x.shape[0]

        # reduce vertically
        query = x.mean(dim=2)  # (N, 32, fW)
        assert torch.isfinite(query).all(), "Invalid values after vertical reduction (mean)"

        # channel last
        query = query.permute(0, 2, 1)  # (N, fW, 32)
        assert torch.isfinite(query).all(), "Invalid values after permuting query"

        # reshape from (N, 32, fH, fW) to (N, 32, fHxfW)
        x = x.view(list(x.shape[:2]) + [-1])  # (N, 32, fHxfW)
        assert torch.isfinite(x).all(), "Invalid values after view reshape"

        # channel last to cope with fc
        x = x.permute(0, 2, 1)  # (N, fHxfW, 32)
        assert torch.isfinite(x).all(), "Invalid values after permuting x"

        # compute 2d positional encoding here
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)  # (fH, fW, 2)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (fH, fW, 32)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))  # (1, fHxfW, 32)
        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
        assert torch.isfinite(pos_enc_2d).all(), "Invalid values in 2D positional encoding"
        x = torch.cat((x, pos_enc_2d), dim=-1)  # (N, fHxfW, 32+32)
        assert torch.isfinite(x).all(), "Invalid values after concatenating 2D positional encoding"

        # get the 1d positional encoding here
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5  # (fW,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))  # (fW, 32)
        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))  # (N, fW, 32)
        assert torch.isfinite(pos_enc_1d).all(), "Invalid values in 1D positional encoding"
        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, fW, 32+32)
        assert torch.isfinite(query).all(), "Invalid values after concatenating 1D positional encoding"

        # attention
        query = self.q_proj(query)  # (N, fW, 32)
        assert torch.isfinite(query).all(), "Invalid values after query projection"
        key = self.k_proj(x)  # (N, fHxfW, 32)
        assert torch.isfinite(key).all(), "Invalid values after key projection"
        value = self.v_proj(x)  # (N, fHxfW, 32)
        assert torch.isfinite(value).all(), "Invalid values after value projection"

        # resize the mask
        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(torch.bool)  # (N, fH, fW)
            mask = torch.logical_not(mask)  # True is not allowed to attend, original mask has True on valid values
            mask = mask.reshape((mask.shape[0], 1, -1))  # (N, 1, fHxfW)
            mask = mask.repeat(1, fW, 1)  # (N, fW, fHxfW)

        x, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, fW, 32)
        assert torch.isfinite(x).all(), "Invalid values after attention"
        
        return x, attn_w












class depth_net_metric3d_old(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_metric3d_old()

    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (
            1 / self.d_hyp
        )  # (D,)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # weighted average
        d = torch.sum(prob * d_vals, dim=-1)  # (N, fW)

        return d, attn, prob



class depth_feature_metric3d_old(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Resnet backbone:
        (480, 640, 3) -> Resnet50(3) -> (30, 40, 1024) # this is the conv feature

        Attn:
        img (480, 640, 3) => ResNet (30, 40, 32) => Avg_Pool (1, 40, 32) ==> Q_proj (1, 40, 32) ==> Attention (1, 40, 32) # this is the attn feature
                            (+ positional encoding)  (+ positional encoding)                             / |
                                            | \=============================> K_proj (30, 40, 32) ======/  |
                                            |===============================> V_proj (30, 40, 32) =========|

        """
        #res50 = resnet50(
        #    pretrained=True, replace_stride_with_dilation=[False, False, True]
        #)
        #self.resnet = nn.Sequential(
        #    IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        #)
        #self.conv = ConvBnReLU(
        #    in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        #)

        self.input_channels=1
        self.initial_filters=32
        self.num_filters=128
        self.num_downsampling_layers=5
        layers = []
        current_channels = self.input_channels
        # Initial convolution to increase channel dimension
        layers.append(nn.Conv2d(self.input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        current_channels = self.initial_filters
        # Downsampling layers with gradual increase in the number of filters
        for _ in range(self.num_downsampling_layers - 1):
            next_channels = current_channels * 2
            layers.append(nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_channels = next_channels        
        # Ensure final channel dimension is 128
        if current_channels != self.num_filters:
            layers.append(nn.Conv2d(current_channels, self.num_filters, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))        
        self.cnn = nn.Sequential(*layers)

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        # Attention block, attention to 2D image
        self.q_proj = nn.Linear(160, 128, bias=False) # 40x40=160
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        #x = self.resnet(x)["feat"]  # (N, 1024, fH, fW)
        #x = self.conv(x)  # (N, 32, fH, fW)
        #print("x.shape: ", x.shape)
        x = self.cnn(x)  # (N, 128, fH, fW)
        fH, fW = list(x.shape[2:])
        #print("fH: ", fH)
        #print("fW: ", fW)
        N = x.shape[0]

        # reduce vertically
        query = x.mean(dim=2)  # (N, 32, fW)

        # channel last
        query = query.permute(0, 2, 1)  # (N, fW, 32)

        # reshape from (N, 32, fH, fW) to (N, 32, fHxfW)
        x = x.view(list(x.shape[:2]) + [-1])  # (N, 32, fHxfW)
        # channel last to cope with fc
        x = x.permute(0, 2, 1)  # (N, fHxfW, 32)

        # compute 2d positional encoding here
        # todo:
        # Example: for (4, 4) image
        # | (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), (-1.5, 1.5)|
        # | (-0.5, -1.5), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5)|
        # | (0.5, -1.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 1.5)|
        # | (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)|
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)  # (fH, fW, 2)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (fH, fW, 32)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))  # (1, fHxfW, 32)
        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
        x = torch.cat((x, pos_enc_2d), dim=-1)  # (N, fHxfW, 32+32)

        # get the 1d positional encoding here
        # todo:
        # Example: for (5, ) ray
        # |-2, -1, 0, 1, 2|
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5  # (fW,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))  # (fW, 32)
        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))  # (N, fW, 32)
        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, fW, 32+32)

        # attention
        query = self.q_proj(query)  # (N, fW, 32)
        key = self.k_proj(x)  # (N, fHxfW, 32)
        value = self.v_proj(x)  # (N, fHxfW, 32)

        # resize the mask
        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
                torch.bool
            )  # (N, fH, fW)
            mask = torch.logical_not(
                mask
            )  # True is not allow to attend, original mask as True on valid values
            mask = mask.reshape((mask.shape[0], 1, -1))  # (N, 1, fHxfW)
            # same mask for all fW
            mask = mask.repeat(1, fW, 1)  # (N, fW, fHxfW)
        x, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, fW, 32)
        #print("x.shape: ", x.shape)
        return x, attn_w