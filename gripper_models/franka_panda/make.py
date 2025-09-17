import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import os


def make_franka_mesh(root):
    hand = trimesh.load(os.path.join(root, 'collision/hand.obj'))
    lf = trimesh.load(os.path.join(root, 'collision/finger.obj'))
    rf = trimesh.load(os.path.join(root, 'collision/finger.obj'))

    # offset
    offset_z = 0.0584
    
    l_offset = [0, 0.015, offset_z]
    r_offset = [0, -0.015, offset_z]
    
    # rotate
    rf_tf = np.eye(4)
    rf_tf[:-1, :-1] = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()#common.euler2rot([0, 0, np.pi])
    rf_tf[:-1, -1] = r_offset
    
    lf_tf = np.eye(4)
    lf_tf[:-1, -1] = l_offset
    
    lf.apply_transform(lf_tf)
    rf.apply_transform(rf_tf)

    combined_hand = trimesh.util.concatenate([hand, rf, lf])
    tf = np.eye(4)
    tf[:-1, -1] = [0, 0, -0.105]
    tf[:-1, :-1] = R.from_euler('xyz', [0, 0, 0.78539816339]).as_matrix()#common.euler2rot([0, 0, 0.785398163397])
    combined_hand.apply_translation([0, 0, -0.105])
    
    return combined_hand


