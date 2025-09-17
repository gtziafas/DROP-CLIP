import os
import numpy as np 
import cv2
import h5py
import math
from PIL import Image
from tqdm import tqdm 
import gc
import torch
import shutil

import utils.transforms as tutils
from utils.geometry import remove_stat_outlier, find_closest_indices, find_existing_points,rgbd_to_pointcloud_o3d,remove_table_mask, aggregate_views_blender_new
from models.features.extractor import ClipExtractor
from data.blender import BlenderDataset
from utils.feature_fusion import MultiviewFeatureFusion

from models.features.extractor import ClipExtractor
from models.features.clip import tokenize

import multiprocessing


def save_dataset_h5py_view(filename, points, colors, labels, hex_ids, feats=None, save_obj_feats=False):
    if save_obj_feats:
        with h5py.File(filename, "w") as f:
            mv_group = f.create_group("multiview")
            mv_group.create_dataset("per_obj", data=feats, dtype='float32')
            mv_group.create_dataset("obj_ids", data=list(range(feats.shape[0])), dtype='uint8')
            mv_group.create_dataset("hex_ids", data=hex_ids, dtype='str')

    else:
        with h5py.File(filename, "w") as f:
            pc_group = f.create_group("pointcloud")
            pc_group.create_dataset("xyz", data=points, dtype='float32')
            pc_group.create_dataset("rgb", data=colors, dtype='float32')
            pc_group.create_dataset("label", data=labels, dtype='uint8')


def save_multiview_dataset_h5py(filename, pc, pc_rgb, pc_label, mv, mv_perobj, obj_ids):
    pointcloud_xyz_path = "xyz"
    pointcloud_rgb_path = "rgb"
    pointcloud_label_path = "label"
    obj_ids_path = ""


    with h5py.File(filename, "w") as f:

        # Point cloud group
        pointcloud_group = f.create_group("pointcloud")
        pointcloud_group.create_dataset("xyz", data=pc, dtype='float32')
        pointcloud_group.create_dataset("rgb", data=pc_rgb, dtype='float32')
        pointcloud_group.create_dataset("label", data=pc_label, dtype='uint8')

        # Multiview group
        mv_group = f.create_group("multiview")
        mv_group.create_dataset("patch", data=mv, dtype='float32')
        mv_group.create_dataset("per_obj", data=mv_perobj, dtype='float32')
        mv_group.create_dataset("obj_ids", data=obj_ids, dtype='uint8')

def save_dataset_h5py(filename, rgb_image, depth_image, segmentation_image,
                      pointcloud_xyz, pointcloud_rgb, pointcloud_label,
                      pointcloud_raw, grasps):
    """
    Saves a dataset containing various data types to an HDF5 file.

    Args:
      filename: Name of the HDF5 file to save.
      rgb_images: List of RGB images (numpy arrays).
      depth_images: List of depth images (numpy arrays).
      segmentation_images: List of segmentation images (numpy arrays).
      pointcloud_xyz: Point cloud data (Nx3 numpy array, where N is the number of points).
      pointcloud_rgb: Point cloud RGB colors (Nx3 numpy array).
      grasps: List of grasp dictionaries. Each dictionary should contain:
              - matrix: 4x4 numpy array representing the grasp pose.
              - label: Integer label for the grasp.
              - score: Grasp score (float).
    """
    # Define dataset paths within the HDF5 file
    rgb_path = "images/rgb"
    depth_path = "images/depth"
    segmentation_path = "images/label"
    pointcloud_xyz_path = "pointcloud/xyz"
    pointcloud_rgb_path = "pointcloud/rgb"
    pointcloud_label_path = "pointcloud/label"
    pointcloud_raw_path = "pointcloud/raw"
    grasp_pose_path = "grasps/poses"
    grasp_label_path = "grasps/labels"
    grasp_score_path = "grasps/scores"
    grasp_idx_path = "grasps/indices"

    with h5py.File(filename, "w") as f:
        # Images group
        images_group = f.create_group("images")
        images_group.create_dataset("rgb", data=rgb_image, dtype='uint8')
        images_group.create_dataset("depth", data=depth_image, dtype='float32')
        images_group.create_dataset("label", data=segmentation_image, dtype='uint8')

        # Point cloud group
        pointcloud_group = f.create_group("pointcloud")
        pointcloud_group.create_dataset("xyz", data=pointcloud_xyz, dtype='float32')
        pointcloud_group.create_dataset("rgb", data=pointcloud_rgb, dtype='float32')
        pointcloud_group.create_dataset("label", data=pointcloud_label, dtype='uint8')
        pointcloud_group.create_dataset("raw", data=pointcloud_raw, dtype='float32')

        # Grasps group
        grasp_group = f.create_group("grasps")
        grasp_group.create_dataset("poses", data=grasps['grasp_poses'], dtype='float32')
        grasp_group.create_dataset("labels", data=grasps['grasp_labels'], dtype='uint8')
        grasp_group.create_dataset("scores", data=grasps['grasp_scores'], dtype='float32')
        grasp_group.create_dataset("indices", data=grasps['grasp_indices'], dtype='int32')
    


def prepare_queries(obj_info, scenario):
    if scenario == "cls":
        # only class names
        return {k:[v['cls_name']] for k,v in obj_info.items()}   

    elif scenario == 'cls+attr':
        # names + attributes in case of ambiguity
        names = {k:[v['cls_name']] for k,v in obj_info.items()}   
        for k, v in obj_info.items():
            if v['concepts'] is not None:
                names[k].extend(v['concepts']['Color'])
                names[k].extend(v['concepts']['Material'])
                names[k].extend(v['concepts']['State'])
                if 'Brand' in v['concepts'] and v['concepts']['Brand'] is not None:
                    if isinstance(v['concepts']['Brand'], str):
                        names[k].append(v['concepts']['Brand'])
                    elif isinstance(v['concepts']['Brand'], list):
                        names[k].extend(v['concepts']['Brand'])
        return names

    elif scenario == 'affordance':
        # only unique afordances
        return {k:v['concepts']['Affordance'] if 'Affordance' in v['concepts'] else [v['cls_name']] for k,v in obj_info.items()}

    elif scenario == 'open':
        # only unique objects, use all descriptions
        condition = {k:v['concepts'] is not None and 'More descriptions' in v['concepts'] for k,v in obj_info.items()}
        all_names = {k:v['concepts']['More descriptions'] if condition[k] else [v['cls_name']] for k,v in obj_info.items()}
        for k, v in all_names.items():
            if obj_info[k]['cls_name'] not in all_names[k]:
                all_names[k].append(obj_info[k]['cls_name']) 
        return all_names

    else:
        raise ValueError(f'Unknown eval scenario {scenario}')


@torch.no_grad()
def preprocess_blender_views(root, models_root, split, chp_folder, CLIP, start, end, batch_size=12, device='cuda', voxel_size=0.02, use_visibility=0, use_similarity=1, include_table=False):
    visual_prompt = 'crop-mask'
    use_sim_kernel = 'open'
    use_kernel_neg = 'scene'

    def consistency_checks(scene):
        ins_to_cls = scene['ins_to_cls']
        if not np.allclose(
            np.asarray(list(set(ins_to_cls.keys()))), np.asarray(list(range(len(ins_to_cls))))):
            return False


        return True

    def _cvt_blender_coord(pts):
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]
        return pts

    #dataset = BlenderDataset(root, models_root, split)
    dataset = BlenderDataset(root, models_root=models_root, split=split, grasp_root='.')

    _ = dataset[int(dataset.scene_ids[0])] # set camera intrinsic
    world_scale = dataset.world_scale
    MVFF = MultiviewFeatureFusion(
            dataset.camera_intrinsic, 
            use_visibility=use_visibility, 
            use_similarity=use_similarity, 
            use_sim_kernel="max", 
            use_obj_prior=1, 
            norm_feat=False, 
            device=device
    )
    scaled_voxel_size = voxel_size * world_scale

    for scene_id_int in tqdm(list(range(int(start),1+int(end)))):
    #for scene_id in tqdm(dataset.scene_ids):
        torch.cuda.empty_cache()
        scene_id = "{:0>6}".format(scene_id_int)
        my_file = os.path.join(chp_folder, f'{scene_id}.h5py')
        if os.path.isfile(my_file):
            print(f'Skipping {scene_id} - already exists in {chp_folder}')
            continue

        if scene_id not in dataset.scene_ids:
            print(f'Skipping {scene_id} - doesnt exist in dataset')
            continue

        try:
            scene = dataset[int(scene_id)]
        except:
            print(f"Skipping {scene_id} - assertion at colors")
            continue
            
        col_to_ins = scene['col_to_ins']
        ins_to_cls = scene['ins_to_cls']

        objects_info = scene["objects_info"]
        obj_info = {k:v for k,v in scene['objects_info'].items() if k>0}

        depths = [samp['depth'] for samp in scene['views'].values()]
        camera_poses = [np.asarray(samp['camera']['world_matrix']).astype(np.float32) for samp in scene['views'].values()]
        images = [samp['rgb'] for samp in scene['views'].values()]
        seg_masks, all_obj_ids_2d = dataset.obtain_seg_info(scene)

        if not consistency_checks(scene):
            print(f"Skipping {scene_id} - not consistent")
            continue

        points, colors, labels = aggregate_views_blender_new(scene, dataset.camera_intrinsic, voxel_size=scaled_voxel_size, depth_trunc=25.0)
        points,colors,labels = remove_table_mask(points,colors,labels)
        
        # if f'{scene_id}' not in os.listdir(chp_folder):
        #     points, colors, labels = aggregate_views_blender_new(scene, dataset.camera_intrinsic, voxel_size=voxel_size)
        #     points,colors,labels = remove_table_mask(points,colors,labels)
        #     print('Aggregation complete, saving...')
        #     os.makedirs(os.path.join(chp_folder, f'{scene_id}'), exist_ok=True)
        #     np.savez(os.path.join(chp_folder, f'{scene_id}/pointcloud.npz'), points=points, colors=colors, labels=labels)
        # else:
        #     chp_path = os.path.join(chp_folder, f'{scene_id}/pointcloud.npz')
        #     try:
        #         chp = np.load(chp_path)
        #     except:
        #         print(f"Skipping {scene_id} - no checkpoint")
        #         continue
        #     points, colors, labels = chp['points'], chp['colors'], chp['labels']

        # make queries
        clip_features = CLIP.extract_obj_prior(
                images, #imgs
                seg_masks, #segs
                obj_ids=all_obj_ids_2d, #obj ids per view
                device=device
            )

        torch.cuda.empty_cache()
        # prompts = {0: ['table'], **prepare_queries(obj_info, eval_scenario)}
        # prompt_embeddings_map = {k: torch.as_tensor(v) for k,v in dataset.prompt_embeddings_map.items()}
        # query_embeddings = torch.stack([prompt_embeddings_map[q].cuda() for q in scene['queries'].values()])
        if use_kernel_neg == 'scene':
            use_queries = {0: ['table'], **prepare_queries(obj_info, use_sim_kernel)}
            tokens = [tokenize(text).to(device) for text in use_queries.values()]
            query_embeddings = [CLIP.model.encode_text(text) for text in tokens]
            query_embeddings = torch.stack([emb.mean(0) for emb in query_embeddings])
        
        elif use_kernel_neg == 'all':
            cls_embeddings = [torch.from_numpy(dataset.cls_embedding_table[k]) for k in scene['ins_to_cls'].values()]
            other_embeddings = [
                torch.from_numpy(dataset.cls_embedding_table[k]) for k in range(len(dataset.id_to_name)) if k not in scene['ins_to_cls'].values()]
            query_embeddings = torch.stack(cls_embeddings + other_embeddings).to(device)
        
        query_embeddings /= query_embeddings.norm(dim=-1, keepdim=True)
        del tokens
        torch.cuda.empty_cache()

        (mv_feats_obj, weight_mask, visibility_mask), (points, colors, labels) = MVFF.fuse(
            points, colors, labels, depths, seg_masks, camera_poses, clip_features, query_embeddings.float(), return_obj=True, device=device)

        # remove nans with queries
        mv_feats_obj = mv_feats_obj.cpu().numpy()
        mv_feats_obj_ids = list(range(mv_feats_obj.shape[0]))

        torch.cuda.empty_cache()

        # remove nans with queries
        for idx in mv_feats_obj_ids:
             if np.any(np.isnan(mv_feats_obj[idx, :])):
                #if idx > 0: # Table will always have Nan embeddings... so just report other object
                #   print(f"Detected Nan object embedding, {scene_id}, {idx}, {q}")
                mv_feats_obj[idx, :] = query_embeddings[idx].cpu().numpy()
        #mv_feats = torch.from_numpy(mv_feats_obj[labels]).to(device)
                
        with h5py.File(os.path.join(chp_folder, f'{scene_id}.h5py'), "w") as hdf:
            mv_group = hdf.create_group("multiview")
            mv_group.create_dataset("per_obj", data=mv_feats_obj, dtype='float32')
            mv_group.create_dataset("obj_ids", data=mv_feats_obj_ids, dtype='uint8')
            mv_group.create_dataset("objects_info", data=str(objects_info))

            pc_group = hdf.create_group('pointcloud')
            pc_group.create_dataset("xyz", data=points, dtype='float32')
            pc_group.create_dataset("rgb", data=colors, dtype='float32')
            pc_group.create_dataset("label", data=labels, dtype='uint8')
            pc_group.create_dataset("vis_mask", data=visibility_mask, dtype='float32')

            print(f'Saved {chp_folder}/{scene_id}.h5py.')

        del clip_features, mv_feats_obj, query_embeddings,points,colors,labels,visibility_mask, weight_mask
        torch.cuda.empty_cache()

        # labels_cls = np.vectorize(lambda x: scene['ins_to_cls'][x])(labels)

        # feat_name = f"mv_feats_{visual_prompt}_{scene_id}.h5py"
        # filename = os.path.join(chp_folder, feat_name)
        # save_dataset_h5py_view(filename, None, None, None, feats=mv_feats_obj.cpu().numpy(), save_obj_feats=True)

        # for v, (rgb, depth, seg, obj_ids, camera_pose) in enumerate(list(zip(images, depths, seg_masks, all_obj_ids_2d, camera_poses))):
        #     valid_m = depth < 25. # Remove background points

        #     pc = rgbd_to_pointcloud_o3d(rgb, depth, dataset.camera_intrinsic)
        #     pts_cam = _cvt_blender_coord(np.asarray(pc.points))
        #     pts_w = tutils.transform_pointcloud_to_world_frame(pts_cam, camera_pose)
        #     col_w = np.array(pc.colors)
        #     lab_w = seg[valid_m].flatten()
            
        #     pts_f, col_f, lab_f = remove_table_mask(pts_w, col_w, lab_w)

        #     mask = find_existing_points(points, pts_f)
            
        #     points_f = points[:][mask==True]
        #     colors_f = colors[:][mask==True]
        #     labels_f = labels[:][mask==True]
    
        #     points_f, ind = remove_stat_outlier(points_f.copy(), n_pts=40, ratio=3.0)
        #     colors_f = colors_f[ind]
        #     labels_f = labels_f[ind]

        #     filename = os.path.join(chp_folder, f'{scene_id}', f'view{v}_pointcloud.h5py')
        #     save_dataset_h5py_view(filename, points_f, colors_f, labels_f, save_obj_feats=False)

        #     #print(f'Done {scene_id}, view {v}')


def preprocess_regrad(cfg, split):
    from data.regrad import RegradDataset
    
    cfg['reference_frame'] = 'world'
    
    dataset = RegradDataset(cfg, split)
    intrinsics = proj.CameraIntrinsics(
        dataset.camera_info['intrinsic'])

    os.makedirs(os.path.join(dataset.root, split, "processed"), exist_ok=True)

    ind = 0
    for scene in tqdm(dataset):
        obj_ids = [x['obj_id'] for x in scene['state']]

        for view in range(1, 10):
            # read data from previous format
            img = scene['views'][view]['image'].copy()
            depth = scene['views'][view]['depth'].copy()
            seg = scene['views'][view]['segm2d'].copy()
            pc = scene['views'][view]['pc_xyz']
            pc_rgb = scene['views'][view]['pc_rgb']
            pc_label = scene['views'][view]['pc_label']
            pc_raw = pc.copy()
            cam_pose = dataset.camera_info['extrinsic'][view]
            
            # filter out table points using 3D segmentation
            table_id = 0
            table_mask = pc_label == table_id
            pc = pc[table_mask==False]
            pc_rgb = pc_rgb[table_mask==False]
            pc_label = pc_label[table_mask==False]

            # establish 3D->2D correspondence
            pc_cam = tutils.transform_pointcloud_to_camera_frame(
                pc, cam_pose)
            mapping = proj.pointcloud_to_pixel(
                proj._cvt_regrad_coord(pc_cam), intrinsics.as_dict) # (M, 2)

            if not np.unique(pc_label).astype(int).tolist() == obj_ids:
                x = [x for x in np.unique(pc_label) if x not in obj_ids]
                obj_ids.extend(x)

            # clean noisy 3D segm points
            pc_valid = None
            obj_ids_2d = np.unique(seg)[1:]
            completed = 0

            for obj in obj_ids_2d:
                # noise 
                if obj not in obj_ids:
                    continue

                # get all 2D pixels for each object
                obj_mask_2d = seg == obj

                # 3D segmentation points
                obj_mask = pc_label == obj

                pt_ids = np.argwhere(obj_mask)
                pixels = mapping[pt_ids].squeeze(1).astype(int)
                
                # keep only those that are in object 2D mask
                ys = np.clip(pixels[:, 1], 0, img.shape[0]-1)
                xs = np.clip(pixels[:, 0], 0, img.shape[1]-1)
                valid = np.argwhere(obj_mask_2d[ys, xs])
                pt_ids_valid = pt_ids[valid].flatten()

                # append to valid pointcloud
                if pc_valid is None:
                    pc_valid = pc[pt_ids_valid]
                    pc_valid_rgb = pc_rgb[pt_ids_valid]
                    pc_valid_label = pc_label[pt_ids_valid]
                else:
                    pc_valid = np.concatenate(
                        [pc_valid, pc[pt_ids_valid]], axis=0)
                    pc_valid_rgb = np.concatenate(
                        [pc_valid_rgb, pc_rgb[pt_ids_valid]], axis=0)
                    pc_valid_label = np.concatenate(
                        [pc_valid_label, pc_label[pt_ids_valid]], axis=0)

                completed += 1

            #assert completed == len(obj_ids), (dataset.scene_ids[ind], view)

            # save 
            scene_id = dataset.scene_ids[ind]
            filename = os.path.join(dataset.root, split, "processed", f"{scene_id}_{view}.h5py")
            save_dataset_h5py(filename, img, depth, seg, pc_valid, pc_valid_rgb,
                pc_valid_label, pc_raw, scene['grasps'][view]
            )

        ind += 1



@torch.no_grad()
def preprocess_regrad_aggr_multiview(cfg, split, CLIP, start, end, device="cuda", batch_size=2, voxel_size=0.01, folder_name="processed_mv"):

    from data.regrad import RegradDataset

    cfg['reference_frame'] = 'world'
    
    dataset = RegradDataset(cfg, split)
    intrinsics = proj.CameraIntrinsics(
        dataset.camera_info['intrinsic'])

    scene_ids = sorted(dataset.scene_ids)

    os.makedirs(os.path.join(dataset.root, split, folder_name), exist_ok=True)

    camera_intrinsic ={
        'fx': intrinsics.as_dict['fx'],
        'fy': intrinsics.as_dict['fy'],
        'cx': 420,
        'cy': 420,
        'width': 840,
        'height': 840,
    }
    camera_poses = [dataset.camera_info['extrinsic'][v] for v in range(1,10)]

    for ind in tqdm(list(range(start, end + 1))):

        scene_id = f"{ind:05d}"
        # some raw files lack view data
        try:
            scene = dataset._load_scene(scene_id)
        except:
            print(f"Skipping {scene_id}, doesnt exist or lacks data.")
            continue

        obj_ids = [x['obj_id'] for x in scene['state']]

        img_input = []
        seg_input = []
        pc_input = []
        pc_rgb_input = []
        pc_label_input = []

        for view in range(1, 10):
            if not scene['views'][view]['valid']:
                continue
            
            # read data from previous format
            img = scene['views'][view]['image'].copy()
            depth = scene['views'][view]['depth'].copy()
            seg = scene['views'][view]['segm2d'].copy()
            pc = scene['views'][view]['pc_xyz']
            pc_rgb = scene['views'][view]['pc_rgb']
            pc_label = scene['views'][view]['pc_label']
            pc_raw = pc.copy()
            cam_pose = dataset.camera_info['extrinsic'][view]
            
            # filter out table points using 3D segmentation
            table_id = 0
            table_mask = pc_label == table_id
            pc = pc[table_mask==False]
            pc_rgb = pc_rgb[table_mask==False]
            pc_label = pc_label[table_mask==False]

            # establish 3D->2D correspondence
            pc_cam = tutils.transform_pointcloud_to_camera_frame(
                pc, cam_pose)
            mapping = proj.pointcloud_to_pixel(
                proj._cvt_regrad_coord(pc_cam), intrinsics.as_dict) # (M, 2)

            if not np.unique(pc_label).astype(int).tolist() == obj_ids:
                x = [x for x in np.unique(pc_label) if x not in obj_ids]
                obj_ids.extend(x)

            # clean noisy 3D segm points
            pc_valid = None
            obj_ids_2d = np.unique(seg)[1:]
            completed = 0

            for obj in obj_ids_2d:
                # noise 
                if obj not in obj_ids:
                    continue

                # get all 2D pixels for each object
                obj_mask_2d = seg == obj

                # 3D segmentation points
                obj_mask = pc_label == obj

                pt_ids = np.argwhere(obj_mask)
                pixels = mapping[pt_ids].squeeze(1).astype(int)

                if len(pixels.shape) < 2:
                    continue
                
                # keep only those that are in object 2D mask
                ys = np.clip(pixels[:, 1], 0, img.shape[0]-1)
                xs = np.clip(pixels[:, 0], 0, img.shape[1]-1)
                valid = np.argwhere(obj_mask_2d[ys, xs])
                pt_ids_valid = pt_ids[valid].flatten()

                # append to valid pointcloud
                if pc_valid is None:
                    pc_valid = pc[pt_ids_valid]
                    pc_valid_rgb = pc_rgb[pt_ids_valid]
                    pc_valid_label = pc_label[pt_ids_valid]
                else:
                    pc_valid = np.concatenate(
                        [pc_valid, pc[pt_ids_valid]], axis=0)
                    pc_valid_rgb = np.concatenate(
                        [pc_valid_rgb, pc_rgb[pt_ids_valid]], axis=0)
                    pc_valid_label = np.concatenate(
                        [pc_valid_label, pc_label[pt_ids_valid]], axis=0)

                completed += 1

            #assert completed == len(obj_ids), (dataset.scene_ids[ind], view)

            img_input.append(img)
            seg_input.append(seg)
            pc_input.append(pc_valid)
            pc_label_input.append(pc_valid_label)
            pc_rgb_input.append(pc_valid_rgb)

        _ = CLIP.set_mode('patch')
        features = CLIP.extract([
            Image.fromarray(im) for im in img_input], device=device)

        mv, pc_aggr = proj.fuse_multiview_features(
                pc_input,
                torch.stack(features).to(device),
                camera_poses,
                camera_intrinsic,
                norm_feat=True,
                reshape_feat=True,
                voxel_size=voxel_size
        )

        pc_aggr_raw = np.concatenate(pc_input, axis=0)
        pc_aggr_rgb_raw = np.concatenate(pc_rgb_input, axis=0)
        pc_aggr_label_raw = np.concatenate(pc_label_input, axis=0)
        ids = find_closest_indices(pc_aggr_raw, pc_aggr)
        pc_aggr_rgb = pc_aggr_rgb_raw[ids]
        pc_aggr_label = pc_aggr_label_raw[ids]

        # change mode to cls
        _ = CLIP.set_mode('cls')
        
        if set(obj_ids) != set(np.unique(pc_aggr_label).tolist()):
            assert set(np.unique(pc_aggr_label).tolist()).issubset(set(obj_ids))
            obj_ids = np.unique(pc_aggr_label).astype(int).tolist()

        features = CLIP.extract_obj_prior(
            img_input,
            seg_input,
            [obj_ids] * len(img_input),
            device=device
        )

        _, _, per_obj_feats = proj.fuse_multiview_features_obj_prior(
            pc_input,
            pc_label_input,
            torch.stack(features).to(device),
            obj_ids,
            voxel_size=voxel_size
        )

        # save 
        filename = os.path.join(dataset.root, split, folder_name, f"{scene_id}.h5py")
        save_multiview_dataset_h5py(filename, pc_aggr, pc_aggr_rgb, pc_aggr_label,
            mv.cpu().numpy(), per_obj_feats.cpu().numpy(), obj_ids
        )
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


def preprocess_wrapper(arg):
    #preprocess_regrad_aggr_multiview(**arg)
    preprocess_blender_views(**arg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', help='dataset to preprocess (REGRAD or Blender)', type=str, default='Blender')
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for inference', type=int, default=9)
    parser.add_argument('-r', '--root', help='data root directory', type=str, default="../temp_data")
    parser.add_argument('-mr', '--models_root', help='object model root directory', type=str, default=None)
    parser.add_argument('-s', '--split', help='which dataset split to process', type=str, default="train")
    parser.add_argument('-f', '--folder_name', help='name of directory to dump processed files', type=str, default="processed_clip")
    parser.add_argument('-p', '--prefix', help='prefix to images directory', type=str, default=None)
    parser.add_argument('-v', '--voxel_size', help='voxel size for downsampling', type=float, default=0.004)
    parser.add_argument('-n', '--n_processes', help='multiprocessing', type=int, default=8)
    parser.add_argument('-start', '--start_idx', help='scene index to start processing from', type=int, default=None)
    parser.add_argument('-end', '--end_idx', help='scene index to end processing', type=int, default=None)

    kwargs = vars(parser.parse_args())

    device = kwargs['device']
    batch_size = kwargs['batch_size']
    root = kwargs['root']
    split = kwargs['split']
    n_processes = kwargs['n_processes']
    voxel_size = kwargs['voxel_size']
        
    if device == 'cuda':
        torch.cuda.init()

    if kwargs['dataset'] == 'REGRAD':
        from utils.config import load_cfg_from_cfg_file
        cfg = load_cfg_from_cfg_file('config/REGRAD.yaml')
        part = kwargs['prefix']
        cfg['root_dir'] = root
        cfg['grasp_dir'] = 'grasp'
        cfg['with_grasp'] = False
        cfg['RGB_dir'] = f'{part}/RGBImages'
        cfg['Depth_dir'] = f'{part}/DepthImages'
        cfg['Seg_dir'] = f'{part}/SegmentationImages'
        folder_name = kwargs['folder_name']

        filenames = sorted(os.listdir(os.path.join(root, split, cfg['RGB_dir'])))
        if kwargs['start_idx'] is None:
            start_from_idx = int(filenames[0].split('.')[0].split('_')[0])
            end_idx = int(filenames[-1].split('.')[0].split('_')[0])

        else:
            start_from_idx = kwargs['start_idx']
            end_idx = kwargs['end_idx']

        print(f'Doing split {split}, part {part}. Saving in {folder_name}')
        
        chunk_size = math.ceil((end_idx - start_from_idx) / n_processes)

        CLIP = ClipExtractor(batch_size=batch_size, device=device, mode='patch')

        args_list = []
        for n in range(n_processes):
            start = start_from_idx + chunk_size * n 
            end = min(start_from_idx + chunk_size * (n+1), end_idx)
            print(f'Process {n+1}: From "{start:05d}" to "{end:05d}"')
            args_list.append({
                'cfg': cfg,
                'split': split,
                'device': device,
                'batch_size': batch_size,
                'voxel_size': voxel_size,
                'folder_name': folder_name,
                'start': start,
                'end': end,
                'CLIP': CLIP
            })

    elif kwargs['dataset'] == 'Blender':

        CLIP = ClipExtractor(
                     model_name="ViT-L/14@336px", 
                     device=device, 
                     mode='cls',
                     visual_prompt = 'crop-mask',
                     crop_num_levels = 1,
                     crop_expansion_ratio = 0.15,
                     blur_kernel=41,
                     batch_size=batch_size, 
                     img_crop=None, img_resize=[336, 448], center_crop=None
        )

        chp_folder = os.path.join(kwargs['root'], kwargs['folder_name'])
        os.makedirs(chp_folder, exist_ok=True)
        start_from_idx = kwargs['start_idx']
        end_idx = kwargs['end_idx']
        chunk_size = math.ceil((end_idx - start_from_idx) / n_processes)
        
        print(f'Doing split {split}, Saving in {chp_folder}')
        print(f'start {start_from_idx}, end {end_idx}, chunk size {chunk_size}')
        
        
        args_list = []
        for n in range(n_processes):
            start = start_from_idx + chunk_size * n 
            end = min(start_from_idx + chunk_size * (n+1), end_idx)
            print(f'Process {n+1}: From "{start:05d}" to "{end:05d}"')
            args_list.append({
                'root': root,
                'models_root': kwargs['models_root'],
                'split': split,
                'device': device,
                'voxel_size': voxel_size,
                'batch_size': batch_size,
                'chp_folder': chp_folder,
                'start': start,
                'end': end,
                'CLIP': CLIP
            })

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(n_processes) as p:
        p.map(preprocess_wrapper, args_list)