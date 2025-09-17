import h5py
import torch
import numpy as np
import random
import json
import os
import MinkowskiEngine as ME

import utils.augmentations as aug 


class MVDistilDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.cfg = cfg
        self.root = cfg.root_dir
        self.split = split
        self.data_dir = os.path.join(self.root, self.split)
        self.feat_key = cfg.feat_key

        # load filepaths and scene IDs
        self.filepaths = sorted([os.path.join(
                self.data_dir, 
                self.cfg.processed_dir, 
                f
            ) for f in  os.listdir(os.path.join(self.data_dir, self.cfg.processed_dir))
        ])

        self.scene_ids = [x.split('.')[0].split('/')[-1] for x in self.filepaths]

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
                [
                    # spatial transforms
                    aug.ElasticDistortion(elastic_distort_params),
                    aug.RandomBlobRemovalPerObj(*blob_removal_params),
                    aug.RandomHorizontalFlip('z', is_temporal=False),
                ]
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

        # refer queries for validation set
        if self.cfg.evaluate and self.split in ["seen_val", "unseen_val"]:
            # # @TODO: Temporary for UNIQUE cls names in each scene, will later change to actual refers 
            print(f'Loading objects from {self.data_dir}/{self.cfg.objects_val_path}')
            self.objects_json = json.load(open(
                f'{self.data_dir}/{self.cfg.objects_val_path}'))
        elif self.split == "train":
            print(f'Loading objects from {self.data_dir}/{self.cfg.objects_train_path}')
            self.objects_json = json.load(open(
                f'{self.data_dir}/{self.cfg.objects_train_path}'))

        self.objectset = self.objects_json['objectset']
        self.objects_json = self.objects_json['scenes']
        #self.scene_ids = sorted(list(self.objects_json.keys()))
        self.scene_ids = sorted(
            list((set(self.scene_ids).intersection(list(self.objects_json.keys())))))
        self.cls_map = json.load(open(
            f'{self.root}/cls_map.json'))

    def load_scene(self, scene_id):
        return h5py.File(os.path.join(
            self.data_dir, self.cfg.processed_dir, f'{scene_id}.h5py'), 'r')

    @staticmethod
    def reconstruct_per_obj_feat(pc, label, feat, obj_ids):
        mv_obj = np.zeros((pc.shape[0], feat.shape[-1]), dtype=float)
        for i, obj in enumerate(obj_ids):
            ids = np.argwhere(label == obj)
            mv_obj[ids, :] = feat[i] 
        return mv_obj

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        scene_id = self.scene_ids[index]
        scene = self.load_scene(scene_id)

        xyz = scene['pointcloud']['xyz'][:]
        rgb = scene['pointcloud']['rgb'][:]
        label = scene['pointcloud']['label'][:]
        obj_ids = scene['multiview']['obj_ids'][:]

        if 'patch' in scene['multiview'].keys() and self.feat_key == "patch":
            feat = scene['multiview']['patch'][:]
        
        elif 'per_obj' in scene['multiview'].keys() and self.feat_key == "per_obj":
            feat = self.reconstruct_per_obj_feat(
                    xyz,
                    label,
                    scene['multiview']['per_obj'][:],
                    obj_ids.tolist()
            )

        else:
            raise ValueError(f"Unknown key {self.feat_key} in dataset keys {scene['multiview'].keys()}")
        feat_dim = feat.shape[-1]

        # center shift
        xyz -= xyz.mean(0)
        if self.use_augm and self.split == "train":
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

            # cat rgb and feats to maintain dimensionality after augmentation
            cat_feat = np.concatenate([rgb, feat], axis=-1) # (M, 3+C)
            xyz, cat_feat, label = self.coord_transforms(xyz, cat_feat, label)
            rgb = cat_feat[:, :3]
            feat = cat_feat[:, 3:]

            if self.cfg.use_color_augmentation:
                rgb_uint8 = (255 * rgb).astype(np.uint8)
                xyz, rgb_uint8, label = self.color_transforms(xyz, rgb_uint8, label)
                rgb = (rgb_uint8 / 255.).astype(np.float32)
                
        xyz = torch.from_numpy(xyz).float()
        rgb = torch.from_numpy(rgb).float()
        feat = torch.from_numpy(feat).float()
        label = torch.from_numpy(label).int()

        cat_features = [feat, xyz, rgb] if self.cfg.use_color else [feat, xyz]
        voxel_coords, voxel_cat_features, voxel_labels, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates = xyz,
            features = torch.cat(cat_features, dim=-1), # (M, C+3+3),
            labels = label,
            ignore_label=0,
            return_index = True,
            return_inverse = True,
            quantization_size = self.cfg.voxel_size
        )
        voxel_features_target = voxel_cat_features[:, :feat_dim] 
        voxel_features_input = voxel_cat_features[:, feat_dim:]
        
        data_dict = {
            'coords': voxel_coords.float(),
            'input_features': voxel_features_input.float(),
            'label': voxel_labels.long(),
            'obj_ids': torch.as_tensor(obj_ids).long(),
            'output_features': voxel_features_target.float(),
            'inverse_map': inverse_map,
            'scene_id': scene_id,
        }

        #recover cls labels / query embeddings for validation
        model_names = {
            x['obj_id'] : x['model_name'] for x in self.objects_json[scene_id]
        }
        obj_names = [model_names[obj] for obj in obj_ids]
        indices = [self.cls_map[name] for name in obj_names]
        label_cls = torch.ones_like(voxel_labels) * 255
        for j, obj in enumerate(obj_ids):
            label_cls[voxel_labels==obj] = indices[j]

        data_dict = {
            **data_dict,
            'label_cls': label_cls.int()
            }

        if self.cfg.evaluate and self.split in ['seen_val', 'unseen_val'] and self.cfg.eval_task in ["all", "grounding"]:
            in_obj_ids = [x['obj_id'] for x in self.objects_json[scene_id] if x['exists']]
            obj_queries = {}
            for obj in obj_ids:
                if obj not in in_obj_ids:
                    continue
                name = model_names[obj]
                if name not in obj_queries:
                    obj_queries[name] = [obj]
                else:
                    obj_queries[name].append(obj)

            data_dict = {
                **data_dict,
                'obj_queries': obj_queries
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
        obj_ids = [data['obj_ids'] for data in batch]
        #queries = [data['queries'] for data in batch]
        raw_labels_cls = [data['label_cls'] for data in batch]

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
        _, labels_cls = ME.utils.sparse_collate(
                    coords=coords,
                    feats=raw_labels_cls,
                ) 

        out_dict = {
            'coords': coordinates,
            'input_features': input_features,
            'output_features': output_features,
            'labels': labels,
            'inverse_map': inverse_maps,
            'scene_ids': scene_ids,
            'obj_ids': obj_ids,
            'labels_cls': labels_cls
        }

        if self.cfg.evaluate and self.split in ['seen_val', 'unseen_val']:
            if self.cfg.eval_task in ["all", "grounding"]:
                obj_queries = [data['obj_queries'] for data in batch]
                out_dict = {
                    **out_dict,
                    'obj_queries': obj_queries
                }

        return out_dict


def build_dataset(args):
    train_data = MVDistilDataset(args, split='train')
    collate_fn = train_data.collate_fn
    if args.evaluate:
        val_data = MVDistilDataset(args, split='seen_val')
        collate_fn = val_data.collate_fn 
        return train_data, val_data, collate_fn
    else:
        return train_data, None, collate_fn