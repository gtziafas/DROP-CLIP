import os
import json
import copy
import cv2
import random
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
from functools import lru_cache

import torch

from utils.geometry import aggregate_views_regrad, find_closest_indices
from utils.grasp import SceneGrasps
import utils.image as imutils
import utils.transforms as tutils
import utils.viz as viz


class RegradDataset(torch.utils.data.Dataset):

    # views alignment pcd data <-> image data
    # VIEWS_MAPPING = {
    #     1: 9,
    #     2: 1,
    #     3: 2,
    #     4: 3,
    #     5: 4,
    #     6: 5,
    #     7: 6,
    #     8: 7,
    #     9: 8
    # }
    VIEWS_MAPPING = {
        1: 9,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8
    }

    PALLETE = viz.PALLETE
    PALLETE_MAP = viz.PALLETE_MAP
    IMAGE_SIZE = (1280, 960)

    def __init__(self, config, split, transform_img=None):
        self.config = config
        self.root = config.root_dir
        self.split = split
        self.data_dir = os.path.join(self.root, self.split)
        self.nviews = config.num_views

        self.image_size = self.IMAGE_SIZE if config.image_resize is None else config.image_resize
        self.transform_img = transform_img

        # # If True, it gives single view data in __getitem__
        # self.as_view = config.as_views

        # If set to "camera", all pointcloud and grasp data for 
        # each view will be expressed in camera local reference frame
        self.reference_frame = self.config.reference_frame

        self._init_data()


    def _init_data(self):
        # load object states
        fname = "objects.json" if self.split == "train" else "objects_16k.json"
        self.objects_json = json.load(open(f'{self.data_dir}/{fname}'))

        # load camera informations
        self.camera_info = np.load(os.path.join(self.root, self.config.camera_file), allow_pickle=True).item()

        #self.scene_ids = sorted(os.listdir(os.path.join(self.root, self.split, self.config.grasp_dir)))
        self.scene_ids  = sorted(next(os.walk(os.path.join(self.root, self.split, self.config.grasp_dir)))[1])

        self.idx_to_view_id = []
        self.idx_to_scene_id = []
        for i, scene_id in enumerate(self.scene_ids):
            self.idx_to_scene_id.append(scene_id)
            for v in range(1, 1 + self.nviews):
                self.idx_to_view_id.append(f"{scene_id}_{v}")

    #@lru_cache(maxsize=None)
    def _load_img(self, scene_id, view):
        fname = f"{scene_id}_{view}.jpg"
        img = Image.open(os.path.join(self.data_dir, self.config.RGB_dir, fname)).convert('RGB')
        
        if self.config.image_resize:
            img = cv2.resize(np.asarray(img), self.image_size, cv2.INTER_CUBIC)
            img = Image.fromarray(img)

        img = self.transform_img(img) if self.transform_img is not None else np.asarray(img)
        
        return img 

    #@lru_cache(maxsize=None)
    def _load_depth(self, scene_id, view):
        fname = f"{scene_id}_{view}.png"
        depth = cv2.imread(os.path.join(self.data_dir, self.config.Depth_dir, fname),
            cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        
        if self.config.image_resize:
            depth = cv2.resize(depth, self.image_size, cv2.INTER_CUBIC)
        
        depth = torch.from_numpy(depth).float() if self.transform_img else depth
        
        return depth

    #@lru_cache(maxsize=None)
    def _load_seg(self, scene_id, view):
        fname = f"{scene_id}_{view}.png"
        seg = cv2.imread(os.path.join(self.data_dir, self.config.Seg_dir, fname), 
            cv2.IMREAD_UNCHANGED)        
        seg[seg >= 200] = 0 # replace white background pixels with 0
        
        if self.config.image_resize:
            # we have to deconstruct to binary masks, resize there and reconstruct
            obj_ids = np.unique(seg)
            binary_masks = imutils.seg_mask_to_binary(seg)
            binary_masks_resized = [cv2.resize(
                m.astype(np.uint8) * 255, self.image_size, cv2.INTER_MAX) for m in binary_masks]
            binary_masks_resized = np.stack(binary_masks_resized).astype(bool)
            seg = imutils.binary_masks_to_seg(binary_masks_resized, obj_ids) 
        
        seg = torch.from_numpy(seg).long() if self.transform_img else seg
        
        return seg

    #@lru_cache(maxsize=None)
    def _load_grasp_data(self, scene_id, view):
        return np.load(f'{self.data_dir}/{self.config.grasp_dir}/{scene_id}/{scene_id}_view_{view}.p', allow_pickle=True)
        
    #@lru_cache(maxsize=None)
    def _load_pc(self, scene_id, view):
        data = self._load_grasp_data(scene_id, view)
        xyz = data['view_cloud']
        rgb = data['view_cloud_color']
        label = data['view_cloud_label'] + 1
        anno = np.array([self.PALLETE_MAP[x+1] for x in data['view_cloud_label']])   
        return xyz, rgb, label, anno, data['scene_cloud'], data['scene_cloud_table']

    #@lru_cache(maxsize=None)
    def _load_grasps(self, scene_id, view):
        data = self._load_grasp_data(scene_id, view)
        
        indices = data['valid_index']

        poses = data['select_frame']

        if self.config.analytical_scores: 
            scores = {
                'center': data['select_center_score'],
                'vertical': data['select_vertical_score'],
                'antipodal': data['select_antipodal_score'],
                'total': data['select_score']
            }
        else:
            scores = data['select_score']

        labels = data['select_frame_label'] + 1

        return indices, poses, scores, labels

    def _load_scene(self, scene_id):
        objs = self.objects_json[scene_id]
        
        filtered_cloud = None

        result = {}
        all_grasps = {}
        
        # state is same accross view besides bbox - read 1st
        state = [{k:v for k,v in o.items() if k not in ["minAreaRect", "bbox"]} for o in objs['1']]
    
        for v in range(1, self.nviews+1):
            _objs = objs[str(v)]
            valid = True

            # missing grasp data
            try:
                xyz, color, label, anno, full_cloud, full_cloud_table = self._load_pc(scene_id, v)
            except:
                valid = False

            # missing views
            try:
                img = self._load_img(scene_id, self.VIEWS_MAPPING[v])
            except:
                valid = False

            if filtered_cloud is None and self.config.include_pc_filtered:
                filtered_cloud = full_cloud
            
            if not valid:
                result[v] = {'valid': False}
                continue

            result[v] = {
                'image': img, 
                'pc_xyz': xyz,
                'pc_label': label,
                'pc_anno': anno,
                'pc_rgb': color,
                'RGB_boxes': {},
                'RGB_rectangles': {},
                '6D_poses': {},
                'valid': True
            }

            if self.config.with_depth:
                depth = self._load_depth(scene_id, self.VIEWS_MAPPING[v])
                result[v] = {**result[v], 'depth': depth}
            if self.config.with_seg:
                seg = self._load_seg(scene_id, self.VIEWS_MAPPING[v])
                result[v] = {**result[v], 'segm2d': seg}
            if self.config.with_grasp:
                indices, poses, scores, labels = self._load_grasps(scene_id, v)
                grasps = {
                    'grasp_indices': indices,
                    'grasp_poses': poses,
                    'grasp_scores': scores.astype(np.float32),
                    'grasp_labels': labels.astype(np.uint8),
                }
                all_grasps[v] = grasps
            
            for j, o in enumerate(_objs):
                cam = self.camera_info['extrinsic'][v]

                # convert object 6D pose to camera frame if desired
                obj_pose = o['6D_pose']
                if self.reference_frame == "camera":
                    pose = np.eye(4)
                    pose[:3, :3] = R.from_quat(obj_pose[3:]).as_matrix()
                    pose[:3, -1] = np.array(obj_pose[:3])
                
                    pose_camera = tutils.transform_poses_to_camera_frame(
                        pose[None, :, :], cam)[0]

                    pos = pose_camera[:3, -1]
                    quat = R.from_matrix(pose_camera[:3, :3]).as_quat()
                    obj_pose = np.concatenate([pos, quat], axis=0)

                result[v]['6D_poses'][o['obj_id']] = obj_pose

                result[v]['RGB_boxes'][o['obj_id']] = objs[str(self.VIEWS_MAPPING[v])][j]['bbox']
                result[v]['RGB_rectangles'][o['obj_id']] = objs[str(self.VIEWS_MAPPING[v])][j]['minAreaRect']

        #pc_aggr = aggregate_views(result)
        pc_aggr = aggregate_views_regrad(
            {k:view for k,view in result.items() if view['valid']==True})
        pc_aggr = {
            'pc_xyz': pc_aggr['xyz'],
            'pc_rgb': pc_aggr['rgb'],
            'pc_label': pc_aggr['label'],
            'pc_anno': pc_aggr['anno'],
        }
        if self.config.include_pc_filtered:
            point_indices = find_closest_indices(pc_aggr['pc_xyz'], 
                filtered_cloud)
        
            pc_aggr = {**pc_aggr, 
                'pc_filt_xyz': pc_aggr['pc_xyz'][point_indices],
                'pc_filt_rgb': pc_aggr['pc_rgb'][point_indices],
                'pc_filt_label': pc_aggr['pc_label'][point_indices],
                'pc_filt_anno': pc_aggr['pc_anno'][point_indices],
            }

        ret = {'views': result, 'aggr': pc_aggr, 'state': state}
        
        if self.config.with_grasp:
            ret = {**ret, 'grasps': all_grasps}

        if self.reference_frame == "camera":
            ret = tutils.transform_scene_to_camera_frame(
                ret, self.camera_info, self.config.with_grasp)
        
        return ret

    def __getitem__(self, index):
        # if self.as_view:
        #     # Each index is one view
        #     scene_id, view = self.idx_to_view_id[index].split('_')
        #     _scene = self._load_scene(scene_id)
        #     scene = {'aggr' : _scene['aggr'],
        #         **_scene['views'][int(view)]}
        # else:
        #     # Each index is one scene
        #     scene_id = self.idx_to_scene_id[index]
        #     scene = self._load_scene(scene_id)
        scene_id = self.idx_to_scene_id[index]
        scene = self._load_scene(scene_id)
        return scene

    def __len__(self):
        #return len(self.idx_to_view_id) if self.as_view else len(self.idx_to_scene_id)
        return len(self.idx_to_scene_id)


    def visualize_scene(self, index, view=0, seg=False, world_f=False, camera_f=False):
        scene = self.__getitem__(index)

        # view = 0 --> aggregated pointcloud
        if view == 0:
            xyz = scene['aggr']['pc_xyz']
            col = scene['aggr']['pc_anno'] if seg else scene['aggr']['pc_rgb']
        # else, view-dependant pointcloud
        else:
            assert view in list(range(1, self.nviews + 1)), f"view must be between 1 - {self.nviews+1}"
            xyz = scene['views'][view]['pc_xyz']
            col = scene['views'][view]['pc_anno'] if seg else scene['views'][view]['pc_rgb']

        more_meshes = []
        if world_f:
            more_meshes.append(viz.get_coord_frame(scale=0.25))

        if camera_f:
            if view > 0:
                ext = self.camera_info['extrinsic'][view]
                more_meshes.append(viz.get_coord_frame(scale=0.25, transform=ext))
            else:
                more_meshes.extend([
                    viz.get_coord_frame(scale=0.25, transform=ext) for ext in self.camera_info['extrinsic'].values()
                ])

        return viz.pcshow(xyz, col, more_meshes)


    def visualize_grasps(self, index, view=0, score_thresh=.75, max_grasps=50, sort=False, use_mesh=True, object_only=None, seg=False):
        scene = self.__getitem__(index)

        # view = 0 --> aggregated grasps
        if view == 0:
            xyz = scene['aggr']['pc_xyz']
            col = scene['aggr']['pc_anno'] if seg else scene['aggr']['pc_rgb']
            
            g_poses = np.zeros((1, 4, 4)).astype(np.float32)
            g_scores = np.zeros((1,)).astype(np.float32)
            g_labels = np.zeros((1,)).astype(np.uint8)
            g_ids = np.zeros((1,)).astype(np.int32)
            
            for v in range(1, 10):
                g_poses = np.concatenate(
                    (g_poses, scene['grasps'][v]['grasp_poses']), axis=0)
                g_scores = np.concatenate(
                    (g_scores, scene['grasps'][v]['grasp_scores']), axis=0)
                g_labels = np.concatenate(
                    (g_labels, scene['grasps'][v]['grasp_labels']), axis=0)
                g_ids = np.concatenate(
                    (g_ids, scene['grasps'][v]['grasp_indices']), axis=0)

            g_poses = g_poses[1:, ...]
            g_scores = g_scores[1:, ...]
            g_labels = g_labels[1:, ...]
            g_ids = g_ids[1:, ...]

            # transform back to world if needed
            if self.reference_frame == "camera":
                g_poses = tutils.transform_poses_to_world_frame(
                    g_poses, self.camera_info['extrinsic'][v])

        else:
            assert view in list(range(1, self.nviews + 1)), f"view must be between 1 - {self.nviews+1}"
            xyz = scene['views'][view]['pc_xyz']
            col = scene['views'][view]['pc_anno'] if seg else scene['views'][view]['pc_rgb']

            g_poses = scene['grasps'][view]['grasp_poses']
            g_scores = scene['grasps'][view]['grasp_scores']
            g_labels = scene['grasps'][view]['grasp_labels']
            g_ids = scene['grasps'][view]['grasp_indices']

        grasps = SceneGrasps(g_ids, g_poses, g_scores, g_labels)
        
        # filter by thresh
        grasps._filter_by_score(score_thresh)

        # filter by obj label
        if object_only is not None:
            grasps._filter_by_labels(object_only)

        if sort:
            # sort by score
            grasps._select_topk(max_grasps)
        else:
            # random sample
            grasps._sample(max_grasps)

        g_meshes = grasps.to_meshes(
            use_gripper_mesh=use_mesh, gripper_type=self.config.gripper_type)

        print(f'Visualizing {len(g_meshes)} grasps for objects {object_only}')

        return viz.pcshow(xyz, col, g_meshes)



REGRAD_OBJECTS_TEST = [
 'airplane ',
 'bag',
 'bench',
 'birdhouse',
 'bottle',
 'camera',
 'cap',
 'car',
 'earphone',
 'guitar',
 'helmet',
 'motorcycle',
 'mug',
 'pistol',
 'remote controller',
 'rocket'
]