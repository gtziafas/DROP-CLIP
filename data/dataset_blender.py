import h5py
import torch
import numpy as np
import random
import json
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import glob
import MinkowskiEngine as ME
import utils.augmentations as aug 
import utils.transforms as tutils
from PIL import Image
import einops
from ast import literal_eval
from collections import Counter


class MVDistilDataset(torch.utils.data.Dataset):
    MAX_POINTS = 10000

    def __init__(self, cfg, split):
        super().__init__()
        self.cfg = cfg
        self.root = cfg.root_dir
        self.split = split
        self.data_dir = os.path.join(self.root, self.split)
        self.feat_key = cfg.feat_key
        self.use_view_clip = cfg.use_view_clip
        self.use_full_pc = cfg.use_full_pc
        self.num_view = 73
        
        # training options: [single full cloud per scene, single partial cloud per scene, many partial clouds per scene]
        h5f_list = glob.glob(os.path.join(self.root, self.split, "*", "*.h5py"))
        self.data = []
        if not cfg.use_full_pc:
            for h5f in h5f_list:
                scene_id = h5f.split("/")[-2]
                if self.cfg.use_k_views > 1:
                    self.data.append([h5f, -1])
                else:
                    assert self.cfg.use_view_ids is not None
                    self.cfg.use_view_ids = list(map(int, self.cfg.use_view_ids.split(',')))
                    for i in self.cfg.use_view_ids:
                        self.data.append([h5f, i])
        else:
            for h5f in h5f_list:
                scene_id = h5f.split("/")[-2]
                self.data.append([h5f, -1])
        
        self.scene_ids = [x for x in os.listdir(f"{self.root}/{self.split}") if os.path.isdir(f"{self.root}/{self.split}/{x}")]

        if self.use_view_clip:
            from models.features.clip import clip
            from models.features.extractor import ClipExtractor
            self.CLIP = ClipExtractor(
                model_name="ViT-L/14@336px", 
                device='cuda',
                mode='patch', 
                batch_size=12, 
                img_crop=None, 
                img_resize=[336, 448], 
                center_crop=None
            )
            self.patch_h = 336//14
            self.patch_w = 448 //14
            
            self.K = np.asarray([
                [444.44444444, 0, 319.5],
                [0, 444.44444444, 239.5],
                [0,  0,  1]
                
            ])
            
        # augmentations in train set
        self.use_augm = cfg.use_augmentation
        if self.use_augm and self.split == "train":
            elastic_distort_params = (
                (
                    cfg.aug_elastic_distortion_granularity_min,
                    cfg.aug_elastic_distortion_granularity_max,
                ),
                (
                    cfg.aug_elastic_distortion_magnitude_min,
                    cfg.aug_elastic_distortion_magnitude_max,
                )
            )

            self.coord_transforms = aug.Compose(
                [
                    # spatial transforms
                    aug.ElasticDistortion(elastic_distort_params),
                    aug.RandomHorizontalFlip('z', is_temporal=False),
                    # random shift & small rotation + add noisy points manually in __getitem__
                ]
            ) 
            if self.cfg.aug_use_blob_removal:
                blob_removal_params = (
                    (
                        cfg.aug_n_blob_min,
                        cfg.aug_n_blob_max,
                    ),
                    (
                        cfg.aug_blob_size_min,
                        cfg.aug_blob_size_max,
                    )
                )
                self.coord_transforms = aug.Compose(
                    self.coord_transforms + \
                    [aug.RandomBlobRemovalPerObj(*blob_removal_params)]
                )

            if cfg.use_color and cfg.use_color_augmentation:
                self.color_transforms = aug.Compose(
                    [   
                        # color transforms
                        aug.ChromaticAutoContrast(),
                        aug.ChromaticTranslation(cfg.aug_color_trans_ratio),
                        aug.ChromaticJitter(cfg.aug_color_trans_ratio),
                        aug.HueSaturationTranslation(
                            cfg.aug_hue_max, cfg.aug_saturation_max)
                    ]
                )

    def load_h5py(self, view_f):
        return h5py.File(view_f, 'r')

    @staticmethod
    def reconstruct_per_obj_feat(pc, label, feat, obj_ids):
        return feat[label]

    @torch.no_grad()
    def generate_view_clip(self, pc, scene_id, view_id, h=480, w=640):
        def _cvt_blender_coord(pts):
            pts[:, 1] = -pts[:, 1]
            pts[:, 2] = -pts[:, 2]
            return pts
        
        projected_points = np.zeros((pc.shape[0], 2), dtype = int)
        
        rgb_f = f"{self.root}/{scene_id}/image.{scene_id}.rgb.view{int(view_id):03d}.png"
        cam_pose = json.load(
            open(f"{self.root}/{scene_id}/cameras.{scene_id}.json", "r")
        )[f"view{int(view_id):03d}"]
        
        points_camera = tutils.transform_pointcloud_to_camera_frame(pc, np.asarray(cam_pose['world_matrix']))
        points_camera = _cvt_blender_coord(points_camera)
        projected_points_not_norm = (self.K @ points_camera.T).T
        mask = (projected_points_not_norm[:, 2] != 0) 
        projected_points[mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
                projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
        
        clip_feature = [einops.rearrange(f, "(h w) c -> h w c", h = self.patch_h, w=self.patch_w) for f in self.CLIP.extract([rgb_f])][0]
        clip_feature = torch.nn.functional.interpolate(
                        clip_feature.permute(2,0,1).unsqueeze(0), 
                        size=(h, w), 
                        mode="bicubic", 
                        align_corners=False
            ).squeeze().permute(1,2,0)
        
        projected_points[:, 1] = np.clip(projected_points[:, 1], 0, h-1)
        projected_points[:, 0] = np.clip(projected_points[:, 0], 0, w-1)
        
        # print(projected_points[:, 1].max(), projected_points[:, 1].min())
        # print(projected_points[:, 0].max(), projected_points[:, 0].min())
        # print(projected_points.shape)
        # print(clip_feature.shape)
        view_clip_features = clip_feature[projected_points[:, 1], projected_points[:, 0]]
        # print(view_clip_features.shape)
        
        return view_clip_features.cpu()

    def prepare_queries(self, obj_info):
        def extract_queries(obj):
            # Extract brand, color, state, and material from queries
            queries = obj['queries']
            brand = queries.get('Brand')
            color = queries.get('Color', [])
            state = queries.get('State', [])
            material = queries.get('Material', [])
            return {
                'brand': brand,
                'color': color,
                'state': state,
                'material': material
            }

        def find_unique_attribute(obj_info):
            # Find unique and non-unique objects
            cls_names = [x['cls_name'] for x in obj_info.values()]
            cls_cnt = Counter(cls_names)
            unique_objs = {k: v for k, v in obj_info.items() if cls_cnt[v['cls_name']] == 1}
            non_unique_objs = {k: v for k, v in obj_info.items() if k not in unique_objs.keys()}

            # Collect non unique objects by their class name
            non_unique_cls = {}
            for obj_id, obj_data in non_unique_objs.items():
                cls_name = obj_data['cls_name']
                if cls_name not in non_unique_cls:
                    non_unique_cls[cls_name] = []
                non_unique_cls[cls_name].append((obj_id, obj_data))

            # Process each non-unique class name
            unique_attributes = {}
            for cls_name, obj_list in non_unique_cls.items():
                obj_attrs = {obj_id: extract_queries(obj_data) for obj_id, obj_data in obj_list}
                for obj_id, attrs in obj_attrs.items():
                    # Determine unique attribute with the given priority (brand, color, state, material)
                    if attrs['brand']:
                        unique_attribute = attrs['brand']
                    else:
                        unique_attribute = None
                        for key in ['color', 'state', 'material']:
                            for value in attrs[key]:
                                is_unique = all(value not in other_attrs[key] for other_obj, other_attrs in obj_attrs.items() if other_obj != obj_id)
                                if is_unique:
                                    unique_attribute = value
                                    break
                            if unique_attribute:
                                break
                    if not unique_attribute:
                        unique_attribute = None
                    unique_attributes[obj_id] = unique_attribute
            return unique_objs, non_unique_objs, unique_attributes

        unique_objs, non_unique_objs, unique_attributes = find_unique_attribute(obj_info)

        if self.cfg.eval_scenario == "cls":
            # only unique names
            return {k:[v['cls_name']] for k,v in unique_objs.items() if k>0}   

        elif self.cfg.eval_scenario == 'cls+attr':
            # names + attributes in case of ambiguity
            names = {k:[v['cls_name']] for k,v in unique_objs.items() if k>0}  
            amb = {k:[v] for k,v in unique_attributes.items() if (v is not None and k>0)}
            return {**names, **amb}

        elif self.cfg.eval_scenario == 'ambiguous':
            # only ambiguous cases, use visual attributes and/or instance names
            return {k:[v] for k,v in unique_attributes.items() if (v is not None and k>0)}

        elif self.cfg.eval_scenario == 'affordance':
            # only unique afordances
            return {k:v['queries']['Affordance'] for k,v in unique_objs.items() if 'Affordance' in v['queries']}

        elif self.cfg.eval_scenario == 'open':
            # only unique objects, use all descriptions
            all_names = {k:v['queries']['More descriptions'] for k,v in unique_objs.items() if 'More descriptions' in v['queries']}
            for k, v in all_names.items():
                if unique_objs[k]['cls_name'] not in all_names[k]:
                    all_names[k].append(unique_objs[k]['cls_name']) 
            return all_names

        else:
            raise ValueError(f'Unknown eval scenario {self.cfg.eval_scenario}')

    def remove_nan_objects(self, labels, obj_feats, obj_ids):
        nan_ids = []
        mask = np.ones_like(labels).astype(bool)
        for i in obj_ids:
            if i == 0:
                continue
            else:
                if np.any(np.isnan(obj_feats[i, :])):
                    nan_ids.append(i)
                    mask = np.logical_and(mask, (labels != i))
        
        return mask, nan_ids
        
        
    def __len__(self):
        return len(self.data)

    def _random_rotation(self, data):
        # Rotation is calculated in radians
        p = np.random.uniform(0, 1)
        if p > self.cfg.aug_random_rot_chance:
            rot_x = np.random.uniform(self.cfg.aug_rotate_min_x, self.cfg.aug_rotate_max_x)
            rot_y = np.random.uniform(self.cfg.aug_rotate_min_y, self.cfg.aug_rotate_max_y)
            rot_z = np.random.uniform(self.cfg.aug_rotate_min_z, self.cfg.aug_rotate_max_z)

            R_x = np.asarray(
                            [[1, 0, 0], [0, np.cos(rot_x), -np.sin(rot_x)], [0, np.sin(rot_x), np.cos(rot_x)]]
                        )
            
            R_y = np.asarray(
                            [[np.cos(rot_y), 0, np.sin(rot_y)], [0, 1, 0], [-np.sin(rot_y), 0, np.cos(rot_y)]]
                        )

            R_z = np.asarray(
                            [[np.cos(rot_z), -np.sin(rot_z), 0], [np.sin(rot_z), np.cos(rot_z), 0], [0, 0, 1]]
                        )

            matrices = [R_x, R_y, R_z]

            if self.cfg.aug_random_euler_order:
                random.shuffle(matrices)
            R = np.matmul(matrices[2], np.matmul(matrices[1], matrices[0]))
            return data @ R.T
        else:
            return data

    def __getitem__(self, index):
        hdf_f, view_id = self.data[index]
        scene_id = hdf_f.split("/")[-2]
        
        data = self.load_h5py(hdf_f)
        xyz, rgb, label = data["pointcloud"]["xyz"][:], data["pointcloud"]["rgb"][:], data["pointcloud"]["label"][:]

        obj_feats, obj_ids = data["multiview"]["per_obj"][:], data["multiview"]["obj_ids"][:]
        obj_info = data["multiview"]["objects_info"][()]
        if isinstance(obj_info, bytes):
            obj_info = obj_info.decode('utf-8')
        obj_info = literal_eval(obj_info)

        queries = self.prepare_queries(obj_info)

        mask, nan_ids = self.remove_nan_objects(label, obj_feats, obj_ids)
        xyz = xyz[mask, :]
        rgb = rgb[mask, :]
        label = label[mask]
        all_ids_in_label = np.unique(label)
        
        for id in nan_ids:
            assert id not in all_ids_in_label
        
        if self.use_view_clip:
            view_feat = self.generate_view_clip(xyz, scene_id, view_id)

        feat = self.reconstruct_per_obj_feat(
                xyz,
                label,
                obj_feats,
                obj_ids.tolist()
        )
        feat_dim = feat.shape[-1]

        if not self.use_full_pc:
            if not self.cfg.use_k_views:
                visibility_mask = data["pointcloud"]["vis_mask"][:].astype(np.uint8).astype(bool)[view_id, :]
            else:
                num_views = random.randint(1, self.cfg.use_k_views)
                tmp = data["pointcloud"]["vis_mask"][:]
                view_ids = np.random.choice(np.arange(tmp.shape[0]), size=num_views, replace=False).astype(int)
                visibility_mask = tmp[view_ids, :]
                visibility_mask = visibility_mask.sum(0).astype(bool)

            xyz = xyz[visibility_mask, :]
            rgb = rgb[visibility_mask, :]
            label = label[visibility_mask].astype(np.uint8)
            feat = feat[visibility_mask, :]

        # Random down sample to balance the workload of each worker
        indices = np.random.choice(
            np.arange(xyz.shape[0]),
            self.MAX_POINTS,
            replace = False if self.MAX_POINTS <= xyz.shape[0] else True
        )
        xyz = xyz[indices, :]
        rgb = rgb[indices, :]
        label = label[indices]
        feat = feat[indices, :]

        # center shift
        xyz -= xyz.mean(0)
        if self.use_augm and self.split == "train":
            # do random shift
            if self.cfg.aug_random_shift:
                try:
                    xyz += (
                        np.random.uniform(xyz.min(0), xyz.max(0))
                        / 2
                    )
                except OverflowError as err:
                    print(xyz)
                    print(xyz.shape)
                    raise err
            
            if self.cfg.aug_random_rotation:
                # do random small rotation on all axes
                xyz = self._random_rotation(xyz)

            # cat rgb and feats to maintain dimensionality after augmentation
            cat_feat = np.concatenate([rgb, feat, view_feat], axis=-1) if self.use_view_clip else np.concatenate([rgb, feat], axis=-1)
            xyz, cat_feat, label = self.coord_transforms(xyz, cat_feat, label)
            rgb = cat_feat[:, :3]
            feat = cat_feat[:, 3:3+feat_dim]
            view_feat = cat_feat[:, -feat_dim:] if self.use_view_clip else None

            if self.cfg.use_color and self.cfg.use_color_augmentation:
                rgb_uint8 = (255 * rgb).astype(np.uint8)
                xyz, rgb_uint8, label = self.color_transforms(xyz, rgb_uint8, label)
                rgb = (rgb_uint8 / 255.).astype(np.float32)
                
        xyz = torch.from_numpy(xyz).float()
        rgb = torch.from_numpy(rgb).float()
        feat = torch.from_numpy(feat).float()
        label = torch.from_numpy(label).int()

        cat_features = [feat, xyz]
        if self.cfg.use_color:
            cat_features.append(rgb)
        if self.cfg.use_view_clip:
            cat_features.append(view_feat)
            
        voxel_coords, voxel_cat_features, voxel_labels, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates = xyz,
            features = torch.cat(cat_features, dim=-1), # (M, C+3+3+C),
            labels = label,
            ignore_label=0,
            return_index = True,
            return_inverse = True,
            quantization_size = self.cfg.voxel_size
        )
        voxel_features_target = voxel_cat_features[:, :feat_dim] 
        voxel_features_input = voxel_cat_features[:, feat_dim:]
        
        data_dict = {
            'xyz': xyz,
            'rgb': rgb,
            'feat': feat,
            'view_feat': view_feat if self.use_view_clip else None,
            'raw_label': label,
            'coords': voxel_coords.float(),
            'input_features': voxel_features_input.float(),
            'label': voxel_labels.long(),
            'obj_ids': torch.as_tensor(obj_ids).long(),
            'output_features': voxel_features_target.float(),
            'inverse_map': inverse_map,
            'scene_id': scene_id,
            'view_id': view_id,
            'queries': queries,
        }
        
        return data_dict
    

    def collate_fn(self, batch):
        out_dict = {}
        coords = [data['coords'] for data in batch]
        raw_feats_input = [data['input_features'] for data in batch]
        raw_labels = [data['label'] for data in batch]
        raw_feat_out = [data['output_features'] for data in batch]
        inverse_maps = [data['inverse_map'] for data in batch]
        scene_ids = [data['scene_id'] for data in batch]
        view_ids = [data['view_id'] for data in batch]
        obj_ids = [data['obj_ids'] for data in batch]
        queries = [data['queries'] for data in batch]

        coordinates, input_features = ME.utils.sparse_collate(
            coords=coords,
            feats=raw_feats_input
        )
        _, labels = ME.utils.sparse_collate(
            coords=coords,
            feats=raw_labels,
        )
        _, output_features = ME.utils.sparse_collate(
            coords=coords,
            feats=raw_feat_out
        )

        out_dict = {
            'coords': coordinates,
            'input_features': input_features,
            'output_features': output_features,
            'labels': labels,
            'inverse_map': inverse_maps,
            'scene_ids': scene_ids,
            'view_ids': view_ids,
            'obj_ids': obj_ids,
            'queries': queries
        }
        
        return out_dict


def build_dataset(args):
    train_data = MVDistilDataset(args, split='train')
    collate_fn = train_data.collate_fn
    if args.evaluate:
        val_data = MVDistilDataset(args, split='test')
        collate_fn = val_data.collate_fn 
        return train_data, val_data, collate_fn
    else:
        return train_data, None, collate_fn