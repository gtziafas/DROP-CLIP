# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import random
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME


def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }


def stack_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = (
        torch.stack([d["coordinates"] for d in list_data]),
        torch.stack([d["features"] for d in list_data]),
        torch.cat([d["label"] for d in list_data]),
    )

    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }


class PointNet(nn.Module):
<<<<<<< HEAD
    def __init__(self, in_channel, out_channel, embedding_channel=1024):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
=======
    def __init__(self, in_channels, out_channels, embedding_channel=1024):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1, bias=False)
>>>>>>> 618f3459d7ddc97ee4379952126e41ff195080c8
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embedding_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embedding_channel)
        self.linear1 = nn.Linear(embedding_channel, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
<<<<<<< HEAD
        self.linear2 = nn.Linear(512, out_channel, bias=True)
=======
        self.linear2 = nn.Linear(512, out_channels, bias=True)
>>>>>>> 618f3459d7ddc97ee4379952126e41ff195080c8

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x




# MinkowskiNet implementation of a pointnet.
#
# This network allows the number of points per batch to be arbitrary. For
# instance batch index 0 could have 500 points, batch index 1 could have 1000
# points.
class MinkPointNetBase(ME.MinkowskiNetwork):
    PLANES = None 
    EMBEDDING_CHANNEL = None

<<<<<<< HEAD
    def __init__(self, in_channel, out_channel, D=3, dropout=0.0):
        ME.MinkowskiNetwork.__init__(self, D)
        self.inplanes = in_channel
=======
    def __init__(self, in_channels, out_channels, D=3, dropout=0.0):
        ME.MinkowskiNetwork.__init__(self, D)
        self.inplanes = in_channels
        embedding_channel = self.EMBEDDING_CHANNEL
>>>>>>> 618f3459d7ddc97ee4379952126e41ff195080c8
        all_dims = [self.inplanes] + list(self.PLANES)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.layers = nn.ModuleList([])
        for ind, (inplane, outplane) in enumerate(in_out):
            self.layers.append(
                self._make_layer(inplane, outplane)
            )

        self.max_pool = ME.MinkowskiGlobalMaxPooling()

        self.linear1 = nn.Sequential(
            ME.MinkowskiLinear(self.PLANES[-1], embedding_channel, bias=False),
            ME.MinkowskiBatchNorm(embedding_channel),
            ME.MinkowskiReLU(),
        )
        self.dp1 = ME.MinkowskiDropout()
<<<<<<< HEAD
        self.linear2 = ME.MinkowskiLinear(embedding_channel, out_channel, bias=True)
=======
        self.linear2 = ME.MinkowskiLinear(embedding_channel, out_channels, bias=True)
>>>>>>> 618f3459d7ddc97ee4379952126e41ff195080c8

    @staticmethod
    def _make_layer(in_planes, out_planes, batchnorm=True):
        return nn.Sequential(
            ME.MinkowskiLinear(in_planes, out_planes, bias=False),
            ME.MinkowskiBatchNorm(out_planes) if batchnorm else nn.Identity(),
            ME.MinkowskiReLU(),
        )

    def forward(self, x):
<<<<<<< HEAD
        for layer in range(self.layers):
=======
        for layer in self.layers:
>>>>>>> 618f3459d7ddc97ee4379952126e41ff195080c8
            x = layer(x)
        x = self.max_pool(x)
        x = self.linear1(x)
        x = self.dp1(x)
        return self.linear2(x).F


class MinkPointNetA(MinkPointNetBase):
    PLANES = (64, 64, 64, 128, 1024)
    EMBEDDING_CHANNEL = 512


class MinkPointNetB(MinkPointNetBase):
    PLANES = (64, 64, 64, 128, 1024)
    EMBEDDING_CHANNEL = 768


class MinkPointNetC(MinkPointNetBase):
    PLANES = (64, 64, 64, 128, 256, 1024)
    EMBEDDING_CHANNEL = 768


<<<<<<< HEAD
def mink_pointnet(in_channel=3, out_channel=20, D=3, arch='MinkPointNetA', dropout=0.0)
    if arch == 'MinkPointNetA':
        return MinkPointNetA(in_channel, out_channel, D, dropout)
    elif arch == 'MinkPointNetB':
        return MinkPointNetB(in_channel, out_channel, D, dropout)
    elif arch == 'MinkPointNetC':
        return MinkPointNetC(in_channel, out_channel, D, dropout)
=======
def mink_pointnet(in_channels=3, out_channels=20, D=3, arch='MinkPointNetA', dropout=0.0):
    if arch == 'MinkPointNetA':
        return MinkPointNetA(in_channels, out_channels, D, dropout)
    elif arch == 'MinkPointNetB':
        return MinkPointNetB(in_channels, out_channels, D, dropout)
    elif arch == 'MinkPointNetC':
        return MinkPointNetC(in_channels, out_channels, D, dropout)
>>>>>>> 618f3459d7ddc97ee4379952126e41ff195080c8
    else:
        raise Exception('architecture not supported yet'.format(arch))