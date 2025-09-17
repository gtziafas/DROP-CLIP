import importlib; import data.blender; importlib.reload(data.blender); from data.blender import BlenderDataset
import utils.geometry; importlib.reload(utils.geometry); from utils.geometry import *
import utils.image as imutils
from utils.projections import apply_pca
from utils.viz import *
import utils.feature_fusion; importlib.reload(utils.feature_fusion); from utils.feature_fusion import MultiviewFeatureFusion
import torch
import einops
from PIL import Image
from models.features.clip import tokenize 
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, intersectionAndUnionGPU, trainMetricPC)
import os
from tqdm import tqdm 
import random

random.seed(19)


def viz_pca(mv_feats, points, colors, labels, trans_factor=15):
    pca = apply_pca(mv_feats.cpu().numpy(), seed=0)
    
    all_labels_filt_color = np.array([PALLETE[x] for x in labels.tolist()])
    p_o3d = to_o3d(points, colors)
    pca_o3d = copy.deepcopy(to_o3d(points, pca)).translate([0, -trans_factor, 0])
    p_label_o3d = copy.deepcopy(to_o3d(points, all_labels_filt_color)).translate([trans_factor, 0, 0])
    return o3d_viewer([p_o3d, pca_o3d, p_label_o3d])

def viz_sims(mv_feats, points, colors, labels_cls, queries, ins_to_cls,  thr=0.7, method="paired", trans_factor=15):
    background = colors.copy() * 0.4
    
    for i, (global_id, text_query) in enumerate(queries.items()):
        # if i == 0:
        #    continue # skip table
        cls_id = ins_to_cls[global_id]
        gt_mask = labels_cls==cls_id
        
        #feats3d_norm = feats3d
        qneg = [x for x in queries.values() if x != text_query]
        pred, sims_norm = CLIP.predict(
            mv_feats.half().cuda(), text_query, qneg, method=method, threshold=thr)
        viz_clip_pred_gt(points, pred.cpu().numpy(), gt_mask, sims_norm.cpu().numpy(), text_query, background, trans_factor=trans_factor)


def obtain_seg_info(scene):
    col_to_ins_dict = scene['col_to_ins']
    seg_masks, all_obj_ids_2d = [], []
    for view_id, stuff in scene['views'].items():
        cls_ids, binary_masks, colors = zip(*stuff['annos'])
        global_ids = [col_to_ins_dict[x]["ins_id"] for x in colors]
        seg_ins_2d = imutils.binary_masks_to_seg(np.stack(binary_masks), np.asarray(global_ids))
        seg_masks.append(seg_ins_2d)
        all_obj_ids_2d.append(global_ids)
    return seg_masks, all_obj_ids_2d



def consistency_checks(scene):
    ins_to_cls = scene['ins_to_cls']
    if not np.allclose(
        np.asarray(list(set(ins_to_cls.keys()))), np.asarray(list(range(len(ins_to_cls))))):
        return False
    return True



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
def main(root, split, chp_folder, device, use_obj_prior, use_similarity, use_visibility, use_sim_kernel, sim_thr, sim_method, sim_negatives, voxel_size, batch_size, eval_scenario, visual_prompt, models_root, grasp_root, clip_num_crop_levels=1, clip_crop_expansion_ratio=0.15, n_views=73):
    dataset = BlenderDataset(root, models_root=models_root, split=split, grasp_root=grasp_root)

    if not use_obj_prior:
        from models.features.extractor import ClipExtractor
        CLIP = ClipExtractor(model_name="ViT-L/14@336px", device=device, mode='patch', batch_size=batch_size, img_crop=None, img_resize=[336, 448], center_crop=None)
        patch_h = 336//14; patch_w = 448 //14

    else:
        from models.features.extractor import ClipExtractor
        CLIP = ClipExtractor(model_name="ViT-L/14@336px", device='cuda', 
                     mode='cls',
                     visual_prompt = visual_prompt.split(','),
                     crop_num_levels = clip_num_crop_levels,
                     crop_expansion_ratio = clip_crop_expansion_ratio,
                     blur_kernel=41,
                     batch_size=batch_size, 
                     img_crop=None, img_resize=[336, 448], center_crop=None
        )
    print(f'CLIP obj-prior={use_obj_prior}, visibility={use_visibility}, similarity={use_similarity}')
    print(f'eval scenario:{eval_scenario}, kernel options:{use_sim_kernel}, inference options:{sim_method, sim_thr, sim_negatives}')

    # mvff params
    _ = dataset[int(dataset.scene_ids[0])] # set camera intrinsic
    world_scale = dataset.world_scale
    MVFF = MultiviewFeatureFusion(
            dataset.camera_intrinsic, 
            use_visibility=use_visibility, 
            use_similarity=use_similarity, 
            use_sim_kernel="max", 
            use_obj_prior=use_obj_prior, 
            norm_feat=False, 
            device=device
    )
    scaled_voxel_size = voxel_size * world_scale
    print(f'World scale: {world_scale}, Voxel size: {voxel_size}, Using scaled voxel size {scaled_voxel_size}.')

    mask_iou_list, mask_prec25_list, mask_prec50_list, mask_prec75_list = [], [], [], []
    use_sim_kernel, use_kernel_neg = use_sim_kernel.split(',')
    if 'crop' in visual_prompt:
        visual_prompt += f'_{clip_num_crop_levels}_{clip_crop_expansion_ratio}'

    for scene_id in tqdm(dataset.scene_ids):
        scene = dataset[int(scene_id)]

        depths = [samp['depth'] for samp in scene['views'].values()]
        camera_poses = [np.asarray(samp['camera']['world_matrix']).astype(np.float32) for samp in scene['views'].values()]
        images = [samp['rgb'] for samp in scene['views'].values()]
        seg_masks, all_obj_ids_2d = dataset.obtain_seg_info(scene)

        obj_info = {k:v for k,v in scene['objects_info'].items() if k>0}

        if not consistency_checks(scene):
            print(f"Skipping {scene_id} - not consistent")
            continue

        if n_views < 73:
            use_views = list(range(0, n_views))
            #use_views = random.sample(list(range(73)), n_views)
            depths = [d for i, d in enumerate(depths) if i in use_views]
            camera_poses = [d for i, d in enumerate(camera_poses) if i in use_views]
            images = [d for i, d in enumerate(images) if i in use_views]
            seg_masks = [d for i, d in enumerate(seg_masks) if i in use_views]
            all_obj_ids_2d = [d for i, d in enumerate(all_obj_ids_2d) if i in use_views]

        pc_name = f"pointcloud.v{int(1000*voxel_size)}e-3.npz.npy"
        chp_path = os.path.join(chp_folder, scene_id, pc_name)
        os.makedirs(os.path.join(chp_folder, f'{scene_id}'), exist_ok=True)
        if pc_name not in os.listdir(os.path.join(chp_folder, f'{scene_id}')):
            points, colors, labels = aggregate_views_blender_new(
                scene, dataset.camera_intrinsic, voxel_size=scaled_voxel_size, depth_trunc=25.0)
            points,colors,labels = remove_table_mask(points,colors,labels)
            np.save(chp_path, {'points':points,'colors':colors,'labels':labels}, allow_pickle=True)
            print(f'Aggregation complete. Saving in {chp_path}')
        else:
            try:
                chp = np.load(chp_path, allow_pickle=True).item();
            except:
                print(f"Skipping {scene_id} - no checkpoint")
                continue
            points, colors, labels = chp['points'], chp['colors'], chp['labels']

        # make queries
        prompts = {0: ['table'], **prepare_queries(obj_info, eval_scenario)}
        with torch.no_grad():
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

        if not use_obj_prior:
            feat_name = f"feats_patch_{scene_id}.p"
            if feat_name not in os.listdir(chp_folder):
                clip_features = CLIP.extract([Image.fromarray(im) for im in images], device)
                clip_features = [einops.rearrange(f, "(h w) c -> h w c", h = patch_h, w=patch_w) for f in clip_features]
                print(f"Saving {os.path.join(chp_folder, feat_name)}")
                torch.save(clip_features, os.path.join(chp_folder, feat_name))
            else:
                clip_features = torch.load(os.path.join(chp_folder, feat_name))

            if n_views < 73:
                clip_features = [x for i, x in enumerate(clip_features) if i in use_views]

            (mv_feats, vis_mask, _), (points, colors, labels) = MVFF.fuse(
                points, colors, labels, depths, seg_masks, camera_poses, clip_features, query_embeddings.float(), device=device)
            labels_cls = np.vectorize(lambda x: scene['ins_to_cls'][x])(labels)

        else:
            
            feat_name = f"feats_cls_{visual_prompt}_{scene_id}.p"
            if feat_name not in os.listdir(chp_folder):
                clip_features = CLIP.extract_obj_prior(
                    images, #imgs
                    seg_masks, #segs
                    obj_ids=all_obj_ids_2d, #obj ids per view
                    device=device
                )
                print(f"Saving {os.path.join(chp_folder, feat_name)}")
                torch.save(clip_features, os.path.join(chp_folder, feat_name))
            else:
                clip_features = torch.load(os.path.join(chp_folder, feat_name))
            
            if n_views < 73:
                clip_features = [x for i, x in enumerate(clip_features) if i in use_views]

            (mv_feats_obj, weight_mask, vis_mask), (points, colors, labels) = MVFF.fuse(
                points, colors, labels, depths, seg_masks, camera_poses, clip_features, query_embeddings.float(), return_obj=True, device=device)
            labels_cls = np.vectorize(lambda x: scene['ins_to_cls'][x])(labels)

            # remove nans with queries
            mv_feats_obj = mv_feats_obj.cpu().numpy()
            mv_feats_obj_ids = list(range(mv_feats_obj.shape[0]))
            for idx, q in zip(mv_feats_obj_ids, prompts.values()):
                if np.any(np.isnan(mv_feats_obj[idx, :])):
                    #if idx > 0: # Table will always have Nan embeddings... so just report other object
                    #   print(f"Detected Nan object embedding, {scene_id}, {idx}, {q}")
                    mv_feats_obj[idx, :] = query_embeddings[idx].cpu().numpy()
            mv_feats = torch.from_numpy(mv_feats_obj[labels]).to(device)

        pred_list, gt_list = [], []
        for obj_id, text_queries in prompts.items():
            if obj_id == 0: # skip table
                continue
            
            if sim_negatives == 'scene':
                #negatives = [x for x in scene['queries'].values() if x != text_query]
                negatives = sum([x for k, x in prompts.items() if k not in [0,obj_id]], [])
            elif sim_negatives == 'all':
                negatives = [x for k, x in dataset.id_to_name.items() if k not in [scene['ins_to_cls'][obj_id],0]]
            elif sim_negatives == 'generic':
                negatives = CLIP.NEGATIVE_PROMPT_GENERIC
            else:
                negatives = None

            for text_query in text_queries:
                pred, sims_norm = CLIP.predict(
                    mv_feats.half().to(device),
                    text_query,
                    qneg=negatives,
                    method=sim_method,
                    threshold=sim_thr,
                    norm_vis_feat=True
                )

                gt = torch.from_numpy(labels == obj_id).to(device)

                pred_list.append(pred)
                gt_list.append(gt)

            torch.cuda.empty_cache()

        iou, (pr25, pr50, pr75) = trainMetricPC(pred_list, gt_list, pr_ious=[0.25, 0.5, 0.75], sigmoid=False)
        mask_iou_list.append(iou)
        mask_prec25_list.append(pr25)
        mask_prec50_list.append(pr50)
        mask_prec75_list.append(pr75)

        torch.cuda.empty_cache()
    
    mean_iou = torch.stack(mask_iou_list).mean().item()
    mean_pr25 = torch.stack(mask_prec25_list).mean().item()
    mean_pr50 = torch.stack(mask_prec50_list).mean().item()
    mean_pr75 = torch.stack(mask_prec75_list).mean().item()

    print(mean_iou, mean_pr25, mean_pr50, mean_pr75)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('--root', help='data root directory', type=str, default="/home/p300488/REGRAD-Ref/data/Blender-new")
    parser.add_argument('--models_root', help='object models root directory', type=str, default="/home/p300488/REGRAD-Ref/SyntheticGraspingDataset/models_processed")
    parser.add_argument('--grasp_root', help='grasp annotations root directory', type=str, default="/home/p300488/REGRAD-Ref/data/Blender-new/output_PC1/grasps/sample_grasps")
    parser.add_argument('--split', help='data split (train, val)', type=str, default="train")
    #parser.add_argument('-s', '--split', help='which dataset split to process', type=str, default="train")
    parser.add_argument('--voxel_size', help='voxel size for downsampling', type=float, default=0.004)
    parser.add_argument('--batch_size', help='batch size for feature extraction', type=int, default=12)
    parser.add_argument('--visual_prompt', help='what visual prompt to use for CLIP obj features (separated by ,)', type=str, default="crop-mask")
    parser.add_argument('--clip_num_crop_levels', help='how many crops to use in CLIP visual prompt (crop methods)', type=int, default=1)
    parser.add_argument('--clip_crop_expansion_ratio', help='expansion ratio for CLIP visual prompt crops (aplied if num_levels>1)', type=float, default=0.15)
    parser.add_argument('--use_obj_prior', help='whether to use obj prior or not', type=int, default=0)
    parser.add_argument('--use_visibility', help='whether to use visibility weight average', type=int, default=0)
    parser.add_argument('--use_similarity', help='whether to use similarity weight average', type=int, default=0)
    parser.add_argument('--use_sim_kernel', help='Use [cls, open] when computing informativeness, how to sample negatives [scene,all]', type=str, default="cls,scene")
    parser.add_argument('--n_views', help='number of views for ablation study (default: 73)', type=int, default=73)
    parser.add_argument('--sim_thr', help='similarity eval computation threshold', type=float, default=0.95)
    parser.add_argument('--sim_method', help='method for similarity eval computation [paired, argmax]', type=str, default="paired")
    parser.add_argument('--sim_negatives', help='negative prompting scheme for eval computation [None, scene, all, generic]', type=str, default="scene")
    parser.add_argument('--eval_scenario', help='scenario to evaluate grounding [cls, open]', type=str, default="cls")
    parser.add_argument('--chp_folder', help='folder to load aggregated point clouds and features', type=str, default='/home/p300488/REGRAD-Ref/data/Blender-temp/checkpoints')
    
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)
