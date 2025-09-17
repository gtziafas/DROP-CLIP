'''3D model for distillation.'''
from collections import OrderedDict
from .minkunet import mink_unet
from torch import nn


def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(arch):
    if arch.startswith('MinkUNet'):
        return mink_unet
    # elif arch.startswith('MinkPointNet'):
    #     return mink_pointnet
    else:
        raise Exception('architecture not supported yet'.format(arch))


class DisNet(nn.Module):
    '''3D Sparse UNet for Distillation.'''
    def __init__(self, cfg=None):
        super(DisNet, self).__init__()
        # MinkowskiNet for 3D point clouds
        in_channels = 3
        if cfg.use_color:
            in_channels += 3
        constructor = constructor3d(cfg.arch_3d)
        net3d = constructor(
            cfg, in_channels=in_channels, out_channels=cfg.feat_dim, D=3)
        self.net3d = net3d

    def forward(self, sparse_3d):
        '''Forward method.'''
        return self.net3d(sparse_3d)