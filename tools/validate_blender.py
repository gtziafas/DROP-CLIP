import argparse
import os
import shutil
import sys
import time
import warnings
import platform
from datetime import datetime
from functools import partial
from collections import OrderedDict



import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import MinkowskiEngine as ME

import utils.config as config
import wandb

from models.distil import DisNet

from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn, MultiEpochsDataLoader)
from models.similarity import ClipSimilarity
import json
from data.dataset_blender import build_dataset
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, intersectionAndUnionGPU, trainMetricPC, poly_learning_rate)
from itertools import combinations
from ast import literal_eval

import pdb


warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise ValueError



@torch.no_grad()
def validate_grounding(CLIP, val_loader, model, args):
    #sim_negatives = sim_negatives or args.sim_negatives
    #threshold = threshold or args.sim_norm_thresh
    #method = method or args.sim_method
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

    if args.sim_negatives == 'all':
        #read cls list
        cls_list = json.load(open(os.path.join(args.root_dir, 'cls_list.json')))

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
        
        if not args.eval_upper_bound:
            out = model(sinput)
            if hasattr(args, 'use_cls_head') and args.use_cls_head:
                out, out_cls = out

            out_batched = torch.split(out, batch_shapes)

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
            loss = dloss
        else:
            out_batched = torch.split(targets, batch_shapes)

            sim_loss_list.append(torch.tensor(0.0))
            loss = torch.tensor(0.0)

        labels_batched = torch.split(labels, batch_shapes)
        targets_batched = torch.split(targets, batch_shapes)

        for output, label, obj_queries, tgt in zip(
            out_batched, labels_batched, data["queries"], targets_batched):
            
            pred_list, gt_list = [], []

            for obj_id, text_queries in obj_queries.items():
                if obj_id == 0: # skip table
                    continue

                # either generic or in-scene negative prompts
                if args.sim_negatives == "generic":
                    negatives = CLIP.NEGATIVE_PROMPT_GENERIC
                elif args.sim_negatives == "scene":
                    #negatives = [obj_queries[x]["cls_name"] for x in obj_queries.keys() if int(x) != obj_id and int(x) != 0]
                    negatives = sum([x for k, x in obj_queries.items() if k not in [0,obj_id]], [])
                elif args.sim_negatives == "no":
                    negatives = None
                elif args.sim_negatives == "all":
                    # assumes obj_queries = ['cls_name']
                    negatives = [x for x in cls_list.values() if x != text_queries[0]]

                for text_query in text_queries:
                    pred, sims_norm = CLIP.predict(
                        output.half(),
                        text_query,
                        qneg=negatives,
                        norm_vis_feat=True,
                        method=args.sim_method,
                        threshold=args.sim_norm_thresh,
                    )

                    gt = torch.zeros_like(label, device=label.device)
                    gt[label == obj_id] = True

                    pred_list.append(pred)
                    gt_list.append(gt)
                    
                    torch.cuda.empty_cache()
            
        iou, (pr25, pr50, pr75) = trainMetricPC(pred_list, gt_list, pr_ious=[0.25, 0.5, 0.75], sigmoid=False)
        mask_iou_list.append(iou)
        mask_prec25_list.append(pr25)
        mask_prec50_list.append(pr50)
        mask_prec75_list.append(pr75)


    torch.cuda.empty_cache()
    
    mean_iou = torch.stack(mask_iou_list).mean()
    mean_pr25 = torch.stack(mask_prec25_list).mean()
    mean_pr50 = torch.stack(mask_prec50_list).mean()
    mean_pr75 = torch.stack(mask_prec75_list).mean()
    mean_sim_loss = torch.stack(sim_loss_list).mean()
    if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
        mean_aux_loss = torch.stack(aux_loss_list).mean()
        mean_total_loss = torch.stack(total_loss_list).mean()
    
    if dist.is_initialized():
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

    head = f"Evaluation Grounding:"
    info = head + mean_sim_loss_str
    if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
        info += mean_aux_loss_str + mean_total_loss_str
    info += mean_iou_str + mean_pr25_str + mean_pr50_str  + mean_pr75_str
    logger.info(info)

    wandb_log_dict = {
        "val_steps": -1,
        "mIoU": mean_iou.item(),
        "Pr@25": mean_pr25.item(),
        "Pr@50": mean_pr50.item(),
        "Pr@75": mean_pr75.item(),
        "DistilLoss": mean_sim_loss.item()
    }
    if hasattr(args, 'use_aux_loss') and args.use_aux_loss:
        wandb_log_dict = {
            **wandb_log_dict,
            "AuxLoss": mean_aux_loss.item(),
            "TotalLoss": mean_total_loss.item(),
        }
    if dist.is_initialized():
        if dist.get_rank() in [-1, 0] and args.use_wandb:
            wandb.log(wandb_log_dict)
    
    torch.cuda.empty_cache()

    return wandb_log_dict


if __name__ == "__main__":
    logger.add("./eval-blender.log")
    args = get_parser()

    #ckpt_list = ["best_val_miou_model.pth"]
    ckpt_path = args.resume
    ckpt_model = ckpt_path.split('/')[-2]

    logger.info("=> loading checkpoint '{}'".format(f"{ckpt_path}"))
    checkpoint = torch.load(
        f"{ckpt_path}", map_location="cpu")
    args.start_epoch = checkpoint['epoch']
    model = DisNet(args)

    new_state_dict = OrderedDict()
    for k,v in checkpoint['state_dict'].items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()

    logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            args.resume, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()

    _, val_data, _ = build_dataset(args)
    val_loader = data.DataLoader(val_data,
                                batch_size=args.batch_size_val,
                                shuffle=False,
                                num_workers=args.workers_val,
                                pin_memory=False,
                                drop_last=False,
                                collate_fn=val_data.collate_fn
    )

    CLIP = ClipSimilarity(device='cuda', method=args.sim_method, threshold=args.sim_norm_thresh)

    logger.info(f"Start eval")
    eval_cfg = f"{ckpt_model}:{args.eval_scenario}:{args.sim_method}:{args.sim_negatives}:{args.sim_norm_thresh}{'(UB)' if args.eval_upper_bound else ''}"
    logger.info(f"Eval config: {eval_cfg}")

    results = validate_grounding(
                    CLIP,
                    val_loader, 
                    model, 
                    args, 
    )

    save_name = os.path.join(args.save_path, eval_cfg + '.json')
    with open(save_name, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"End eval")