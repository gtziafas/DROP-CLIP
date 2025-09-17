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

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F

import MinkowskiEngine as ME

from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, intersectionAndUnionGPU, trainMetricPC)
from data.dataset import build_dataset
from models.similarity import ClipSimilarity


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


@torch.no_grad()
def validate_segmentation(val_loader, epoch, args):
    torch.backends.cudnn.enabled = False

    dloss_meter = AverageMeter("Distil Loss", ":2.4f", index=0)
    xloss_meter = AverageMeter("X-Entropy Loss", ":2.4f", index=1)
    intersection_meter = AverageMeter("Intersection3D", ":2.2f", index=2)
    union_meter = AverageMeter("Union3D", ":2.2f", index=3)
    target_meter = AverageMeter("Target3D", ":2.2f", index=4)

    def _get_similarity(vis_feat, txt_feat):
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        return vis_feat @ txt_feat.T
    
    # retrieve class embeddings for dataset
    query, cls_map = obtain_cls_embedding(args)
    query = query.cuda(non_blocking=True)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

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
        
        out_batched = torch.split(targets, batch_shapes)
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

    if args.multiprocessing_distributed:
        if dist.get_rank() in [-1, 0] and args.use_wandb:
            wandb.log(wandb_log_dict)
    
    return wandb_log_dict



@torch.no_grad()
def validate_grounding(val_loader, epoch, args):
    torch.backends.cudnn.enabled = False

    sim_loss_list = []
    # ce_loss_list = []
    mask_iou_list = []
    mask_prec25_list = []
    mask_prec50_list = []
    mask_prec75_list = []

    #CLIP, _ = clip.load("ViT-L/14@336px", device="cuda", jit=False)
    CLIP = ClipSimilarity(device='cuda', 
        method=args.sim_method, threshold=args.sim_norm_thresh)

    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        #obj_ids = [x.cuda(non_blocking=True) for x in data["obj_ids"]]
        labels = to_tensor(data["labels"]).cuda(non_blocking=True)
        targets = to_tensor(data["output_features"]).cuda(non_blocking=True)

        sinput = ME.SparseTensor(
                    coordinates=to_tensor(data["coords"]),
                    features=to_tensor(data["input_features"]),
                    device="cuda"
                ).float()
        batch_shapes = [x.shape[0] for x in sinput.decomposed_features]
        
        out = targets[:]
        out_batched = torch.split(targets, batch_shapes)
        labels_batched = torch.split(labels, batch_shapes)
        targets_batched = torch.split(targets, batch_shapes)

        if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
            loss = (1 - torch.nn.CosineSimilarity()
                    (out, targets)).mean()
        elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
            loss = torch.nn.L1Loss()(out, targets)
        else:
            raise NotImplementedError

        sim_loss_list.append(loss)

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
    # mean_ce_loss = torch.stack(ce_loss_list).mean()
    
    if args.multiprocessing_distributed:
        dist.barrier()
        dist.all_reduce(mean_iou)
        dist.all_reduce(mean_pr25)
        dist.all_reduce(mean_pr50)
        dist.all_reduce(mean_pr75)
        dist.all_reduce(mean_sim_loss)
        #dist.all_reduce(mean_ce_loss)

        mean_iou = mean_iou / dist.get_world_size()
        mean_pr25 = mean_pr25 / dist.get_world_size()
        mean_pr50 = mean_pr50 / dist.get_world_size()
        mean_pr75 = mean_pr75 / dist.get_world_size()
        mean_sim_loss = mean_sim_loss / dist.get_world_size()
        #mean_ce_loss = mean_ce_loss / dist.get_world_size()
        
    mean_iou_str = f"  mIoU: {mean_iou.item():.2f}"
    mean_pr25_str = f"  Pr@25: {mean_pr25.item():.2f}"
    mean_pr50_str = f"  Pr@50: {mean_pr50.item():.2f}"
    mean_pr75_str = f"  Pr@75: {mean_pr75.item():.2f}"
    mean_sim_loss_str = f"  SimLoss: {mean_sim_loss.item():.4f}"
    #mean_ce_loss_str = f"  CELoss: {mean_ce_loss.item():.4f}"
    head = f"Evaluation Grounding: Epoch=[{epoch}/{args.epochs}]"
    info = head + mean_sim_loss_str + mean_iou_str + mean_pr25_str + mean_pr50_str  + mean_pr75_str
    #info = head + mean_iou_str + mean_pr50_str + mean_sim_loss_str + mean_ce_loss_str
    logger.info(info)

    wandb_log_dict = {
        "val_steps": epoch,
        "mIoU": mean_iou.item(),
        "Pr@25": mean_pr25.item(),
        "Pr@50": mean_pr50.item(),
        "Pr@75": mean_pr75.item(),
        "SimLoss": mean_sim_loss.item(),
        #"CELoss": mean_ce_loss.item()
    }

    if args.multiprocessing_distributed:
        if dist.get_rank() in [-1, 0] and args.use_wandb:
            wandb.log(wandb_log_dict)
    
    del CLIP
    torch.cuda.empty_cache()

    return wandb_log_dict


def validate(val_loader, epoch, args):
    if args.eval_task == "segmentation":
        metrics = validate_segmentation(val_loader, epoch, args)
    elif args.eval_task == "grounding":
        metrics =  validate_grounding(val_loader, epoch, args)
    elif args.eval_task == "all":
        metrics1 = validate_segmentation(val_loader, epoch, args)
        metrics2 =  validate_grounding(val_loader, epoch, args)
        metrics = {'segm': metrics1, 'ground': metrics2}
    else:
        raise ValueError(f"Unknown option {args.eval_task}. Please select ['segmentation', 'grounding', 'all'].")

    return metrics


if __name__ == "__main__":
    from utils.config import load_cfg_from_cfg_file
    cfg = load_cfg_from_cfg_file('config/DistilREGRAD.yaml')
    _, val_data, collate_fn = build_dataset(cfg)
    val_loader = torch.utils.data.DataLoader(val_data,
        shuffle=False, collate_fn=collate_fn, batch_size=1)
    cfg['multiprocessing_distributed'] = False
    metrics = validate(val_loader, epoch=0, args=cfg)
    print(metrics)