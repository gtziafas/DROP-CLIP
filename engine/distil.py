import os
import cv2
import time
import json
import gc
import wandb
import numpy as np
import open3d as o3d
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
import h5py

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F

import MinkowskiEngine as ME

from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, intersectionAndUnionGPU, trainMetricPC, poly_learning_rate)
from models.similarity import ClipSimilarity
#from models.distil.loss import SupervisedContrastiveLoss
from utils.viz import *
from utils.projections import apply_pca


def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise ValueError


def obtain_cls_embedding(args):
    if args.dataset == "DistilREGRAD":
        print(f'Loading class embeddings from {args.root_dir}/cls_embedding_table.npy')
        cls_map = json.load(open(
            f'{args.root_dir}/cls_map.json'))
        cls_emb_table = torch.from_numpy(np.load(
            f'{args.root_dir}/cls_embedding_table.npy')).float()

    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

    return cls_emb_table, cls_map 


def batch_aux_hinge_loss(feature_list, label_list, margin=0.05):
    batch_margin_loss = 0.0
    batch_pos_loss = 0.0
    for features, labels in zip(feature_list, label_list):
        # Normalize features for cos sim computation
        features = F.normalize(features, p=2, dim=-1)

        unique_labels = torch.unique(labels)
        K = len(unique_labels)

        masks = labels.unsqueeze(0) == unique_labels.unsqueeze(1)
        mean_features = torch.matmul(masks.float(), features) / masks.sum(1, keepdim=True).float()

        K_mask = ~F.one_hot(torch.arange(0,K)).to(labels.device).bool()

        scene_margin_loss = 0.0
        scene_pos_loss = 0.0

        for k in range(K):
            # Positive samples
            mask_features = features[masks[k]]
            cos_sim = torch.mm(mask_features, mask_features.t())
            pos_cos_sim = cos_sim.mean()

            # Convert to cosine distance
            scene_pos_loss += 1.0 - pos_cos_sim
            
            # Negative samples - take mean from other labels
            other_features = K_mask[k].unsqueeze(1) * mean_features
            mask_features_tile = mask_features.unsqueeze(1)
            other_features_tile = other_features.unsqueeze(0)
            neg_cos_sim = F.cosine_similarity(
                mask_features_tile, other_features_tile, dim=2).mean()
            
            # Apply margin to the loss
            scene_margin_loss += torch.clip(
             -pos_cos_sim + neg_cos_sim + margin, 0)

        batch_margin_loss += scene_margin_loss / K
        batch_pos_loss += scene_pos_loss / K

    batch_margin_loss /= len(feature_list)  # Normalize by number of scenes in the batch
    batch_pos_loss /= len(feature_list)

    return batch_pos_loss, batch_margin_loss


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    '''Distillation pipeline.'''

    torch.backends.cudnn.enabled = True
    meter_dict = {
        "batch_time": AverageMeter('Batch', ':2.2f', index=0),
        "data_time": AverageMeter('Data', ':2.2f', index=1),
        "lr": AverageMeter('Lr', ':1.6f', index=2),
        "distil_loss": AverageMeter('DistilLoss', ':2.4f', index=3),
    }
    if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
        meter_dict = {
            **meter_dict,
            "aux_loss": AverageMeter('AuxLoss', ':2.4f', index=4),
            "total_loss": AverageMeter('TotalLoss', ':2.4f', index=5)
        }
    elif hasattr(args, 'use_cls_head') and args.use_cls_head:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
        meter_dict = {
            **meter_dict,
            "aux_loss": AverageMeter('AuxLoss', ':2.4f', index=4),
            "total_loss": AverageMeter('TotalLoss', ':2.4f', index=5)
        }
    
    progress = ProgressMeter(
        len(train_loader),
        sorted([v for k, v in meter_dict.items()], key=lambda x: x.index),
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)

    end = time.time()
    iters = len(train_loader)
    max_iter = args.epochs * len(train_loader)

    for i, data in enumerate(train_loader):
        meter_dict["data_time"].update(time.time() - end)

        targets = to_tensor(data["output_features"]).cuda(non_blocking=True)
        labels_cls = to_tensor(data["labels_cls"]).cuda(non_blocking=True)
        labels = to_tensor(data["labels"]).cuda(non_blocking=True)

        sinput = ME.SparseTensor(
                    coordinates=to_tensor(data["coords"]),
                    features=to_tensor(data["input_features"]),
                    device="cuda"
                ).float()
        batch_shapes = [x.shape[0] for x in sinput.decomposed_features]

        with amp.autocast(enabled=args.amp):
            out = model(sinput)
            if hasattr(args, 'use_cls_head') and args.use_cls_head:
                out, out_cls = out

            if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
                dloss = (1 - torch.nn.CosineSimilarity()
                        (out, targets)).mean()
            elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
                dloss = torch.nn.L1Loss()(out, targets)
            else:
                raise NotImplementedError
            
            meter_dict["distil_loss"].update(dloss.item())
            
            if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
                out_batched = torch.split(out, batch_shapes)
                labels_batched = torch.split(labels, batch_shapes)
                targets_batched = torch.split(targets, batch_shapes)

                # model outputs
                aux_losses = batch_aux_hinge_loss(
                    out_batched, labels_batched)

                # baseline from targets
                with torch.no_grad():
                    aux_losses_base = batch_aux_hinge_loss(
                        targets_batched, labels_batched)
              
                aux_loss = aux_losses[0] + torch.clip(
                    aux_losses[1] - aux_losses_base[1], 0)
                aux_loss *= args.loss_weight_aux
                
                loss = dloss + aux_loss

                meter_dict['aux_loss'].update(aux_loss.item())
                meter_dict['total_loss'].update(loss.item())

            elif hasattr(args, 'use_cls_head') and args.use_cls_head:
                xloss = criterion(out_cls, labels_cls.long())
                xloss *= args.loss_weight_cls
                loss = dloss + xloss
                meter_dict['aux_loss'].update(xloss.item())
                meter_dict['total_loss'].update(loss.item())

            else:
                loss = dloss 

            # backward
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            if getattr(args, "max_norm", 0.0):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + i/iters)
        
        # adjust learning rate
        # current_iter = epoch * len(train_loader) + i + 1
        # current_lr = poly_learning_rate(
        #     args.base_lr, current_iter, max_iter, power=args.power)

        # for index in range(0, args.index_split):
        #     optimizer.param_groups[index]['lr'] = current_lr
        # for index in range(args.index_split, len(optimizer.param_groups)):
        #     optimizer.param_groups[index]['lr'] = current_lr * 10

        meter_dict["lr"].update(scheduler.get_last_lr()[-1])
        #meter_dict["lr"].update(current_lr)
        meter_dict["batch_time"].update(time.time() - end)

        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0] and args.use_wandb:
                log_dict = {}
                for k, v in meter_dict.items():
                    log_dict[f"training/{k}"] = v.val
                wandb.log(log_dict, step=epoch * len(train_loader) + (i + 1))



@torch.no_grad()
def validate_segmentation(val_loader, model, epoch, args):
    torch.backends.cudnn.enabled = False

    dloss_meter = AverageMeter("Distil Loss", ":2.4f", index=0)
    xloss_meter = AverageMeter("X-Entropy Loss", ":2.4f", index=1)
    intersection_meter = AverageMeter("Intersection3D", ":2.2f", index=2)
    union_meter = AverageMeter("Union3D", ":2.2f", index=3)
    target_meter = AverageMeter("Target3D", ":2.2f", index=4)

    def _get_similarity(vis_feat, txt_feat):
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        return vis_feat @ txt_feat.T
    
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # retrieve class embeddings for dataset
    query, cls_map = obtain_cls_embedding(args)
    query = query.cuda(non_blocking=True)

    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        obj_ids = [x.cuda(non_blocking=True) for x in data["obj_ids"]]
        # queries = [x.cuda(non_blocking=True) for x in data["queries"]]
        labels = to_tensor(data["labels"]).cuda(non_blocking=True)
        labels_cls = to_tensor(data["labels_cls"]).cuda(non_blocking=True)
        targets = to_tensor(data["output_features"]).cuda(non_blocking=True)

        sinput = ME.SparseTensor(
                    coordinates=to_tensor(data["coords"]),
                    features=to_tensor(data["input_features"]),
                    device="cuda"
                ).float()
        batch_shapes = [x.shape[0] for x in sinput.decomposed_features]
        
        out = model(sinput)
        out_batched = torch.split(out, batch_shapes)
        labels_batched = torch.split(labels, batch_shapes)
        labels_cls_batched = torch.split(labels_cls, batch_shapes)
        targets_batched = torch.split(targets, batch_shapes)

        for output, label, objmap, tgt, label_cls in zip(
            out_batched, labels_batched, obj_ids, targets_batched, labels_cls_batched):
            # output: (M, C)
            # query: (K, C)
            # obj_ids: {1, 2, ..., K}
            mask = label!=0
            output = output[mask]
            tgt = tgt[mask]
            label = label[mask]
            label_cls = label_cls[mask]
            # label_inv = torch.as_tensor(
            #     [torch.argwhere(objmap==l.item()).item() for l in label], device=label.device)

            sims = _get_similarity(output, query.float()) # (M, K)
            pred = torch.max(sims, 1)[1] # (M, {1, 2, ..., K})

            # distil Loss
            if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
                dloss = (1 - torch.nn.CosineSimilarity()
                        (output, tgt)).mean()
            elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
                dloss = torch.nn.L1Loss()(output, tgt)
            else:
                raise NotImplementedError

            # cross-entropy loss
            xloss = criterion(sims, label_cls.long())

            intersection, union, target = intersectionAndUnionGPU(
                pred, label_cls.detach(), args.n_classes, args.ignore_label)

            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(
                    union), dist.all_reduce(target)

            intersection, union, target = intersection.cpu(
            ).numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(
                union), target_meter.update(target)

            dloss_meter.update(dloss.item(), args.batch_size)
            xloss_meter.update(xloss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    dloss = dloss_meter.avg 
    xloss = xloss_meter.avg
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    mean_iou_str = f"  mIoU: {mIoU*100:.2f}"
    mean_acc_str = f"  mAcc: {mAcc*100:.2f}"
    mean_sim_loss_str = f"  SimLoss: {dloss:.4f}"
    mean_ce_loss_str = f"  CELoss: {xloss:.4f}"
    head = f"Evaluation Segmentation: Epoch=[{epoch}/{args.epochs}]"
    log = head + mean_sim_loss_str + mean_ce_loss_str + mean_iou_str + mean_acc_str 
    logger.info(log)

    wandb_log_dict = {
        "val_steps": epoch,
        "mIoU": mIoU,
        "mAcc": mAcc.item(),
        "SimLoss": dloss,
        "CELoss": xloss
    }

    if dist.get_rank() in [-1, 0] and args.use_wandb:
        wandb.log(wandb_log_dict)
    
    return wandb_log_dict



@torch.no_grad()
def validate_grounding(val_loader, model, epoch, args):
    torch.backends.cudnn.enabled = False

    sim_loss_list = []
    if args.use_aux_loss:
        aux_loss_list = []
        total_loss_list = []
    mask_iou_list = []
    mask_prec25_list = []
    mask_prec50_list = []
    mask_prec75_list = []

    if hasattr(args, 'use_cls_head') and args.use_cls_head:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    model.eval()
    time.sleep(2)

    #CLIP, _ = clip.load("ViT-L/14@336px", device="cuda", jit=False)
    CLIP = ClipSimilarity(device='cuda', 
        method=args.sim_method, threshold=args.sim_norm_thresh)

    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        #obj_ids = [x.cuda(non_blocking=True) for x in data["obj_ids"]]
        labels = to_tensor(data["labels"]).cuda(non_blocking=True)
        labels_cls = to_tensor(data['labels_cls']).cuda(non_blocking=True)
        targets = to_tensor(data["output_features"]).cuda(non_blocking=True)

        sinput = ME.SparseTensor(
                    coordinates=to_tensor(data["coords"]),
                    features=to_tensor(data["input_features"]),
                    device="cuda"
                ).float()
        batch_shapes = [x.shape[0] for x in sinput.decomposed_features]
        
        out = model(sinput)
        if hasattr(args, 'use_cls_head') and args.use_cls_head:
            out, out_cls = out

        out_batched = torch.split(out, batch_shapes)
        labels_batched = torch.split(labels, batch_shapes)
        targets_batched = torch.split(targets, batch_shapes)

        if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
            out_norm = F.normalize(out, p=2, dim=1)
            targets_norm = F.normalize(targets, p=2, dim=1)
            cos_sim = F.cosine_similarity(
                out_norm, targets_norm, dim=-1)
            dloss = (1 - cos_sim).mean()
        elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
            dloss = torch.nn.L1Loss()(out, targets)
        else:
            raise NotImplementedError
        
        sim_loss_list.append(dloss)
        
        if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
            # model outputs
            aux_losses = batch_aux_hinge_loss(
                out_batched, labels_batched)

            # baseline from targets
            with torch.no_grad():
                aux_losses_base = batch_aux_hinge_loss(
                    targets_batched, labels_batched)
          
            aux_loss = aux_losses[0] + torch.clip(
                aux_losses[1] - aux_losses_base[1], 0)
            aux_loss *= args.loss_weight_aux
            
            loss = dloss + aux_loss
            aux_loss_list.append(aux_loss)
            total_loss_list.append(loss)

        elif hasattr(args, 'use_cls_head') and args.use_cls_head:
            xloss = criterion(out_cls, labels_cls.long())
            xloss *= args.loss_weight_cls
            loss = dloss + xloss
            aux_loss_list.append(xloss)
            total_loss_list.append(loss)

        else:
            loss = dloss

        for output, label, obj_queries, tgt in zip(
            out_batched, labels_batched, data["obj_queries"], targets_batched):
            
            pred_list, gt_list = [], []
            for text_query, obj_ids in obj_queries.items():
                # either generic or in-scene negative prompts
                if args.sim_negatives == "generic":
                    negatives = []
                elif args.sim_negatives == "scene":
                    negatives = [x for x in list(obj_queries.keys()) if x != text_query]

                pred, sims_norm = CLIP.predict(
                    output.half(),
                    text_query,
                    negatives
                )

                gt = torch.zeros_like(label, device=label.device)
                for obj in obj_ids:
                    gt[label == obj] = True

                pred_list.append(pred)
                gt_list.append(gt)

        iou, (pr25, pr50, pr75) = trainMetricPC(pred_list, gt_list, pr_ious=[0.25, 0.5, 0.75], sigmoid=False)
        mask_iou_list.append(iou)
        mask_prec25_list.append(pr25)
        mask_prec50_list.append(pr50)
        mask_prec75_list.append(pr75)
    
    mean_iou = torch.stack(mask_iou_list).mean()
    mean_pr25 = torch.stack(mask_prec25_list).mean()
    mean_pr50 = torch.stack(mask_prec50_list).mean()
    mean_pr75 = torch.stack(mask_prec75_list).mean()
    mean_sim_loss = torch.stack(sim_loss_list).mean()
    if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
        mean_aux_loss = torch.stack(aux_loss_list).mean()
        mean_total_loss = torch.stack(total_loss_list).mean()
    
    if args.multiprocessing_distributed:
        dist.barrier()
        dist.all_reduce(mean_iou)
        dist.all_reduce(mean_pr25)
        dist.all_reduce(mean_pr50)
        dist.all_reduce(mean_pr75)
        dist.all_reduce(mean_sim_loss)
        if args.use_aux_loss:
            dist.all_reduce(mean_aux_loss)
            dist.all_reduce(mean_total_loss)

        mean_iou = mean_iou / dist.get_world_size()
        mean_pr25 = mean_pr25 / dist.get_world_size()
        mean_pr50 = mean_pr50 / dist.get_world_size()
        mean_pr75 = mean_pr75 / dist.get_world_size()
        mean_sim_loss = mean_sim_loss / dist.get_world_size()
        if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
            mean_aux_loss = mean_aux_loss / dist.get_world_size()
            mean_total_loss = mean_total_loss / dist.get_world_size()

    mean_iou_str = f"  mIoU: {mean_iou.item():.2f}"
    mean_pr25_str = f"  Pr@25: {mean_pr25.item():.2f}"
    mean_pr50_str = f"  Pr@50: {mean_pr50.item():.2f}"
    mean_pr75_str = f"  Pr@75: {mean_pr75.item():.2f}"
    mean_sim_loss_str = f"  DistilLoss: {mean_sim_loss.item():.4f}"
    if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
        mean_aux_loss_str = f"  AuxLoss: {mean_aux_loss.item():.4f}"
        mean_total_loss_str = f"    TotalLoss: {mean_total_loss.item():.4f}"

    head = f"Evaluation Grounding: Epoch=[{epoch}/{args.epochs}]"
    info = head + mean_sim_loss_str
    if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
        info += mean_aux_loss_str + mean_total_loss_str
    info += mean_iou_str + mean_pr25_str + mean_pr50_str  + mean_pr75_str
    logger.info(info)

    wandb_log_dict = {
        "val_steps": epoch,
        "mIoU": mean_iou.item(),
        "Pr@25": mean_pr25.item(),
        "Pr@50": mean_pr50.item(),
        "Pr@75": mean_pr75.item(),
        "DistilLoss": mean_sim_loss.item(),
    }
    if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
        wandb_log_dict = {
            **wandb_log_dict,
            "AuxLoss": mean_aux_loss.item(),
            "TotalLoss": mean_total_loss.item(),
        }

    if dist.get_rank() in [-1, 0] and args.use_wandb:
        wandb.log(wandb_log_dict)
    
    del CLIP
    torch.cuda.empty_cache()

    return wandb_log_dict


def validate(val_loader, model, epoch, args):
    if args.eval_task == "segmentation":
        metrics = validate_segmentation(val_loader, model, epoch, args)
    elif args.eval_task == "grounding":
        metrics =  validate_grounding(val_loader, model, epoch, args)
    elif args.eval_task == "all":
        metrics1 = validate_segmentation(val_loader, model, epoch, args)
        metrics2 =  validate_grounding(val_loader, model, epoch, args)
        metrics = {**metrics1, **metrics2}
    else:
        raise ValueError(f"Unknown option {args.eval_task}. Please select ['segmentation', 'grounding', 'all'].")

    return metrics


@torch.no_grad()
def visualization(dataset, model, epoch, args):
    def merge_pointcloud(pc_list):
        points = np.concatenate([np.asarray(pc.points) for pc in pc_list], axis=0)
        colors = np.concatenate([np.asarray(pc.colors) for pc in pc_list], axis=0)

        pcd = to_o3d(points, colors)
        return pcd

    torch.backends.cudnn.enabled = False
    model.eval()
    time.sleep(2)

    #CLIP, _ = clip.load("ViT-L/14@336px", device="cuda", jit=False)
    
    tgt_dir = os.path.join(args.output_dir, "vis", f"epoch-{epoch}/rank-{dist.get_rank()}")
    os.makedirs(tgt_dir, exist_ok=True)

    vis_idx = np.random.randint(len(dataset), size=1)[0]
    data = dataset.collate_fn([dataset[vis_idx]])

    targets = to_tensor(data["output_features"]).cuda(non_blocking=True)
    labels = data["labels"].cuda(non_blocking=True)
    #obj_names = data["obj_names"][0]
    inv_map = data["inverse_map"][0]

    raw_pc = data["input_features"][:, :3].cpu().numpy().astype(np.float32)
    raw_colors = (data["input_features"][:, 3:]).cpu().numpy().astype(np.float32)
    sinput = ME.SparseTensor(
            coordinates=to_tensor(data["coords"]),
            features=to_tensor(data["input_features"]),
            device="cuda"
        ).float()
    with amp.autocast(enabled=args.amp):
        out = model(sinput)
    
    with h5py.File(f"{tgt_dir}/outputs.pcd", "w") as hdf:
        hdf.create_dataset("raw_pc", data=raw_pc)
        hdf.create_dataset("raw_rgb", data=raw_colors)
        hdf.create_dataset("outputs", data=out.cpu().numpy()[inv_map, :])
        hdf.create_dataset("targets", data=targets.cpu().numpy()[inv_map, :])

    #text_features = prepare_text_features(CLIP, obj_names, device=out.device)

    label_color = PALLETE[labels.cpu().numpy().astype(int), :][inv_map, :]
    
    offset = 0.5

    raw_pcd = to_o3d(raw_pc, raw_colors)
    raw_pcd_anno = to_o3d(raw_pc, label_color).translate([offset, 0, 0])
    raw_pcd_inp_feat = to_o3d(raw_pc, apply_pca(targets.cpu().numpy()[inv_map, :])).translate([offset*2, 0, 0])
    raw_pcd_oup_feat = to_o3d(raw_pc, apply_pca(out.cpu().numpy()[inv_map, :])).translate([offset*3, 0, 0])
    pcd = merge_pointcloud([raw_pcd, raw_pcd_anno, raw_pcd_inp_feat, raw_pcd_oup_feat])
    o3d.io.write_point_cloud(f"{tgt_dir}/outputs.pcd", pcd)

    # for k in text_features.keys():
    #     obj_name = text_features[k]["name"]
    #     text_embedding = text_features[k]["text_embedding"].to(out.device)
    #     gt_mask = torch.zeros_like(labels).bool().to(out.device)
    #     for i in text_features[k]["ids"]:
    #         gt_mask = torch.logical_or(gt_mask, (labels==i))
        
    #     gt_mask = gt_mask.cpu().numpy()[inv_map]
    #     gt_colors = raw_colors.copy()
    #     gt_colors[gt_mask, :] = np.asarray([1.0, 0.0, 0.0])
    #     gt_pcd = to_o3d(raw_pc, gt_colors)

    #     pred_mask = out.half() @ text_embedding.t().squeeze()
    #     if pred_mask.max() != pred_mask.min():
    #         pred_mask_norm = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
    #     else:
    #         pred_mask_norm = pred_mask / (pred_mask.max() + 1e-6)
        
    #     pred_mask_norm = pred_mask_norm.float().cpu().numpy()[inv_map]

    #     gt_colors = raw_colors.copy()
    #     gt_colors[gt_mask, :] = np.asarray([1.0, 0.0, 0.0])
    #     cmap = plt.get_cmap("turbo")
    #     heatmap = cmap(pred_mask_norm)[:,:3]
    #     pcd_list = [
    #         to_o3d(raw_pc, gt_colors),
    #         to_o3d(raw_pc, heatmap).translate([offset, 0, 0])
    #     ]
    #     translation = np.asarray([offset, 0, 0])
    #     for idx, thres in enumerate([0.5, 0.75, 0.95]):
    #         pred_mask_bin = (pred_mask_norm > thres)
    #         pred_colors = raw_colors.copy()
    #         pred_colors[pred_mask_bin, :] = np.asarray([1.0, 0.0, 0.0])
    #         pcd_list.append(to_o3d(raw_pc, pred_colors).translate(translation * (idx+2)))
        
    #     with h5py.File(f"{tgt_dir}/{obj_name[0].replace(' ', '-')}-results.h5", "w") as hdf:
    #         hdf.create_dataset("similarity", data=pred_mask_norm)
    #         hdf.create_dataset("gt_mask", data=gt_mask)

    #     pcd = merge_pointcloud(pcd_list)
    #     o3d.io.write_point_cloud(f"{tgt_dir}/{obj_name[0].replace(' ', '-')}-segment.pcd", pcd)

    #del CLIP
    torch.cuda.empty_cache()