import numpy as np
import copy
import torch

def transform_scene_to_camera_frame(scene, camera_info, with_grasp=False):
    scene_t = copy.deepcopy(scene)
    for view in range(1, 10):
        pc = scene_t['views'][view]['pc_xyz']
        extrinsics = camera_info['extrinsic'][view]
        pc_t = transform_pointcloud_to_camera_frame(pc, extrinsics)
        scene_t['views'][view]['pc_xyz'] = pc_t
        if with_grasp:
            grasp_poses = scene_t['grasps'][view]['grasp_poses']
            grasp_poses_t = transform_poses_to_camera_frame(grasp_poses, extrinsics)
            scene_t['grasps'][view]['grasp_poses'] = np.array(grasp_poses_t)
    return scene_t


# def transform_pointcloud_to_camera_frame(pointcloud, extrinsics, reverse_z = False):
#     # if reverse_z:
#     #     pointcloud[:, -1] = - pointcloud[:, -1]

#     # Adding a column of ones to the pointcloud to make it homogeneous
#     ones = np.ones((pointcloud.shape[0], 1))
#     homogeneous_pointcloud = np.hstack((pointcloud, ones))

#     # Invert the camera pose matrix
#     inv_camera_pose = np.linalg.norm(extrinsics)

#     # Transforming the pointcloud to the camera frame using the extrinsic matrix
#     pointcloud_camera_frame = homogeneous_pointcloud.dot(inv_camera_pose.T)

#     # Correcting the z-axis by multiplying it by -1
#     # if reverse_z:
#     #     pointcloud_camera_frame[:, 2] = -pointcloud_camera_frame[:, 2]

#     # Removing the homogeneous coordinate
#     pointcloud_camera_frame = pointcloud_camera_frame[:, :3]

#     return pointcloud_camera_frame


def transform_pointcloud_to_world_frame(pointcloud, camera_pose):
    # Transform each point in the point cloud to world frame
    transformed_pointcloud = np.dot(camera_pose, 
                                    np.vstack([pointcloud.T, np.ones((1, pointcloud.shape[0]))]))
    transformed_pointcloud = transformed_pointcloud[:3, :].T

    return transformed_pointcloud

    
def transform_pointcloud_to_camera_frame(pointcloud, camera_pose):
    # Invert the camera pose matrix
    inv_camera_pose = np.linalg.inv(camera_pose)

    # Transform each point in the point cloud
    transformed_pointcloud = np.dot(inv_camera_pose, 
        np.vstack([pointcloud.T, np.ones((1, pointcloud.shape[0]))]))
    transformed_pointcloud = transformed_pointcloud[:3, :].T

    return transformed_pointcloud


def transform_poses_to_camera_frame(poses, camera_pose):
    # Invert the camera pose matrix
    inv_camera_pose = np.linalg.inv(camera_pose)

    # Extract rotation and translation from the grasp poses
    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3]

    # Combine rotation and translation into homogeneous transformation matrices
    pose_transforms = np.eye(4, dtype=np.float64)[np.newaxis, :, :].repeat(poses.shape[0], 0)  # Initialize with identity
    pose_transforms[:, :3, :3] = rotations
    pose_transforms[:, :3, 3] = translations

    # Transform grasp poses to camera frame
    transformed_poses = np.matmul(inv_camera_pose, pose_transforms)

    return transformed_poses


def transform_poses_to_world_frame(poses, camera_pose):
    # Extract rotation and translation from the poses
    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3]

    # Combine rotation and translation into homogeneous transformation matrices
    pose_transforms = np.eye(4, dtype=np.float64)[np.newaxis, :, :].repeat(poses.shape[0], 0)  # Initialize with identity
    pose_transforms[:, :3, :3] = rotations
    pose_transforms[:, :3, 3] = translations

    # Transform poses from camera frame to world frame
    transformed_poses = np.matmul(camera_pose, pose_transforms)

    return transformed_poses


class CoordTransform2d:
    def __init__(
        self,
        img_dim,
        patch_size,
        resize_dim = None,
    ):
        self.height, self.width = img_dim
        self.crop_size = resize_dim or img_dim
        self.patch_size = patch_size
        self.patch_h = self.crop_size[0] / patch_size
        self.patch_w = self.crop_size[1] / patch_size

    @staticmethod
    def _transform(x, y, scale_h, scale_w):
        x = (x * scale_w).long()
        y = (y * scale_h).long()
        return x, y

    def img_to_patch(self, x, y):
        scale_h = self.patch_h / self.height
        scale_w = self.patch_w / self.width
        return self._transform(x, y, scale_h, scale_w)

    def patch_to_img(self, x, y):
        scale_h = self.height / self.patch_h
        scale_w = self.width / self.patch_w
        return self._transform(x, y, scale_h, scale_w)

    def crop_to_patch(self, x, y):
        scale_h = self.patch_h / self.crop_size[0]
        scale_w = self.patch_w / self.crop_size[1]
        return self._transform(x, y, scale_h, scale_w)
    
    def patch_to_crop(self, x, y):
        scale_h = self.crop_size[0] / self.patch_h
        scale_w = self.crop_size[1] / self.patch_w
        return self._transform(x, y, scale_h, scale_w)

    def img_to_crop(self, x, y):
        scale_h = self.crop_size[0] / self.height
        scale_w = self.crop_size[1] / self.width
        return self._transform(x, y, scale_h, scale_w)

    def crop_to_img(self, x, y):
        scale_h = self.height / self.crop_size[0]
        scale_w = self.width / self.crop_size[1]
        return self._transform(x, y, scale_h, scale_w)


def reconstruct_feature_map(feat, image_shape):
    H, W, _ = image_shape
    patch_h, patch_w, C = feat.shape
    scale_h, scale_w = patch_h / H, patch_w / W

    # Create a grid of coordinates in the original image
    y = torch.arange(H).unsqueeze(1).expand(H, W).float()  # Shape (H, W)
    x = torch.arange(W).unsqueeze(0).expand(H, W).float()  # Shape (H, W)

    # Scale the grid according to the feature map
    y_ = (y * scale_h).long()
    x_ = (x * scale_w).long()

    # Use the scaled coordinates to index into the feature map
    reconstructed = feat[y_, x_]

    return reconstructed