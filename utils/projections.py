import numpy as np 
import open3d as o3d
import torch
import torch.nn.functional as F
import math
import einops
import torchvision.transforms as tv
from PIL import Image
from sklearn.decomposition import PCA
from dataclasses import dataclass

import utils.transforms as tutils
from utils.geometry import pc_voxel_down, find_closest_indices


class CameraIntrinsics:
    def __init__(self, mat):
        # 3x3 camera intrinsic array
        self.fx = mat[0, 0]
        self.fy = mat[1, 1]
        self.cx = mat[0, 2]
        self.cy = mat[1, 2]
        
    def __iter__(self):
        return iter([self.fx, self.fy, self.cx, self.cy])

    @property
    def as_matrix(self):
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy ,self.xy],
            [0, 0, 1.],
        ])

    @property
    def as_dict(self):
        return {'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy}
    
    

def rgbd_to_pointcloud_o3d(rgb, depth, camera_intrinsics):
    intr = o3d.camera.PinholeCameraIntrinsic(
        width=rgb.shape[1], height=rgb.shape[0], 
        fx=camera_intrinsics['fx'],
        fy=camera_intrinsics['fy'],
        cx=camera_intrinsics['cx'],
        cy=camera_intrinsics['cy'],
    )
    rgb_image = o3d.geometry.Image(rgb)
    depth_image = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, depth_scale=1.0, depth_trunc=3, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intr)
    return pcd


def pointcloud_to_pixel(pointcloud, camera_intrinsics):
    # Project 3D points to 2D pixel coordinates (expected expressed in camera frame)
    pixels = np.zeros((pointcloud.shape[0], 2))
    pixels[:, 0] = camera_intrinsics['fx'] * pointcloud[:, 0] / pointcloud[:, 2] + camera_intrinsics['cx']  # x' = fx * x / z + cx
    pixels[:, 1] = camera_intrinsics['fy'] * pointcloud[:, 1] / pointcloud[:, 2] + camera_intrinsics['cy']  # y' = fy * y / z + cy
    return pixels


def depth_to_pointcloud(depth_image, camera_intrinsics):
    # Get the shape of the depth image
    height, width = depth_image.shape

    # Create a grid of coordinates corresponding to the indices of the depth image
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Normalize the x and y coordinates using the camera intrinsics
    x = (u - camera_intrinsics['cx']) / camera_intrinsics['fx']
    y = (v - camera_intrinsics['cy']) / camera_intrinsics['fy']

    # Unproject the normalized coordinates to 3D space using the depth image
    z = depth_image.copy()  # Assuming depth is in millimeters, convert to meters
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    # Stack the coordinates to create the point cloud
    pointcloud = np.stack((x, y, z), axis=-1)

    return pointcloud


def _cvt_regrad_coord(pts):
    pts[:, 2] = -pts[:, 2]
    pts[:, 1] = -pts[:, 1]
    return pts


def _cvt_blender_coord(pts):
    pts[:, 2] = -pts[:, 2]
    return pts


def apply_pca(features,  norm=True, seed=42):
    pca = PCA(n_components=3, random_state=seed)
    X = pca.fit_transform(features)
    if norm:
        X = (X - X.min()) / (X.max() - X.min())
    return X


def project_2d_features_to_3d(
    depth_image, 
    features, 
    camera_intrinsics, 
    center_crop = None,
    transform_to_world = False,
    transform_coords = _cvt_regrad_coord,
    subsample_step = 1,
    camera_extrinsics = None
):
    # crop a region around the center of the image frame
    if center_crop:
        trans = tv.CenterCrop([center_crop, center_crop])
        depth_image = np.array(trans(Image.fromarray(depth_image)))

        # crop features if not already aligned
        if depth_image.shape[0:2] != features.shape[0:2]:
            features = torch.from_numpy(features).permute(2, 0, 1)
            features = trans(features).permute(1, 2, 0).numpy()

    # project depth to 3D
    pc_img = depth_to_pointcloud(depth_image, camera_intrinsics)
    pc = pc_img.reshape(-1, 3) 
    features = features.reshape(-1, features.shape[-1])

    # REGRAD implicit camera frame
    if transform_coords is not None:
        pc = transform_coords(pc)

    # subsample if desired
    if subsample_step is not None:
        pc = pc[::subsample_step, ...]
        features = features[::subsample_step, ...]

    if transform_to_world:
        assert camera_extrinsics is not None
        pc = tutils.transform_pointcloud_to_world_frame(
            pc, camera_extrinsics)

    return pc, features



def fuse_multiview_features(
        pcs, 
        multiview_features, # (V, h, w, C)
        camera_poses, 
        camera_intrinsic, 
        crop_size=336, 
        patch_size=14, 
        voxel_size=0.0075,
        reshape_feat=False,
        norm_feat=True
    ):
    pc_aggr = np.concatenate(pcs, axis=0)
    pc_aggr = pc_voxel_down(pc_aggr, voxel_size)

    n_pts = pc_aggr.shape[0]
    feat_size = multiview_features.shape[-1]
    patch_h = patch_w = crop_size // patch_size

    sum_features = torch.zeros((n_pts, feat_size), dtype=float, device=multiview_features.device)
    counter = torch.zeros((n_pts, 1), dtype=float, device=multiview_features.device)
    
    for pc, feat, camera_pose in zip(pcs, multiview_features, camera_poses):
        pc_aggr_ids, pc_ids = np.unique(find_closest_indices(pc_aggr, pc), return_index=True)

        pc_cam = tutils.transform_pointcloud_to_camera_frame(
            pc, camera_pose)
        mapping = pointcloud_to_pixel(
            _cvt_regrad_coord(pc_cam), camera_intrinsic) # (M, 2)
        
        #pixels = mapping.squeeze().astype(int)

        pixels = mapping[pc_ids].squeeze().astype(int)
        if len(pixels.shape) < 2:
            # noise
            continue
        ys = np.clip(pixels[:, 1], 0, camera_intrinsic['height']-1)
        xs = np.clip(pixels[:, 0], 0, camera_intrinsic['width']-1)
    
        if reshape_feat:
            feat = einops.rearrange(feat, "(h w) c -> h w c", h=patch_h, w=patch_w)
        
        if norm_feat:
            feat /= feat.norm(dim=-1, keepdim=True)

        feat = tutils.reconstruct_feature_map(feat, 
            (camera_intrinsic['height'], camera_intrinsic['width'], 3))
        # feat = F.interpolate(feat, size=(camera_intrinsic['height'], camera_intrinsic['width']),
        #     mode="bicubic", align_corners=True)

        #feat_3d = feat[pixels[:,1], pixels[:,0], :]
        feat_3d = feat[ys, xs]

        #pc_aggr_ids = find_closest_indices(pc_aggr, pc)
        sum_features[pc_aggr_ids, :] = sum_features[pc_aggr_ids, :] + feat_3d
        counter[pc_aggr_ids, :] += 1

    counter[counter==0] = 1e-5
    sum_features = sum_features / counter
    #sum_features = torch.nan_to_num(sum_features, 0)

    return sum_features, pc_aggr


def fuse_multiview_features_obj_prior(
    pcs,
    pcs_label,
    multiview_features, # (K, C)
    obj_map,
    voxel_size=0.0075,
    ):
    pc_aggr_raw = np.concatenate(pcs, axis=0)
    pc_aggr_label_raw = np.concatenate(pcs_label, axis=0)

    pc_aggr = pc_voxel_down(pc_aggr_raw, voxel_size)
    #pc_aggr = pc_aggr_raw.copy()
    ids = find_closest_indices(pc_aggr_raw, pc_aggr)
    pc_aggr_label = pc_aggr_label_raw[ids]

    n_pts = pc_aggr.shape[0]
    feat_size = multiview_features.shape[-1]

    sum_features = torch.zeros((n_pts, feat_size), dtype=torch.half, device=multiview_features.device)
    per_obj_features = []
    for i, obj in enumerate(obj_map):
        pt_ids = np.argwhere(pc_aggr_label == obj)
        feat = torch.stack([f[i] for f in multiview_features], dim=0).mean(0)
        sum_features[pt_ids, :] = feat
        per_obj_features.append(feat)
    per_obj_features = torch.stack(per_obj_features, dim=0)

    return sum_features, pc_aggr, per_obj_features



def pool_multiview_features(aggr_pc, aggr_features):
     # Find the unique points and their indices
    unique_points, inverse_indices = np.unique(aggr_pc, axis=0, return_inverse=True)
    
    # Sort the features according to the inverse indices
    sorted_features = aggr_features[inverse_indices.argsort()]
    
    # Find the count of occurrences for each unique point
    unique_counts = np.bincount(inverse_indices)
    
    # Cumulative sum to get the indices for max pooling
    indices = np.r_[0, np.cumsum(unique_counts)]
    
    # Max-pool the feature vectors for each unique point
    pooled_features = np.maximum.reduceat(sorted_features, indices[:-1], axis=0)
    
    return unique_points, pooled_features
