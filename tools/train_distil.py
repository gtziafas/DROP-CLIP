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

# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = '99ee90fdefff711f21b8b40a0fac1bdb95da2aa5'


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
from data.dataset import build_dataset
from engine.distil import train, validate, validate_grounding, validate_segmentation, visualization
from models.distil import DisNet
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn, MultiEpochsDataLoader)

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


@logger.catch
def main():
    torch.multiprocessing.set_start_method('spawn')
    
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ), join=True)
    
    # children = []
    # for i in range(args.world_size):
    #     subproc = mp.Process(target=main_worker, args=(i, args))
    #     children.append(subproc)
    #     subproc.start()

    # for i in range(args.world_size):
    #     children[i].join()

def main_worker(gpu, args):
    now = datetime.now()
    date_time_string = now.strftime('%d-%m-%Y-%H:%M')
    exp_name = f"Distill-{args.dataset}/{date_time_string}"
    args.output_dir = os.path.join("exp", exp_name)

    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

    # wandb
    if args.rank == 0 and args.use_wandb:
        project = f"Distill-{platform.node()}"
        exp_name = f"Distill-{args.dataset}"
        tag_list = [
            args.dataset,
        ]
        wandb.init(job_type="training",
                   mode="online",
                   config=args,
                   project=project,
                   name=exp_name,
                   tags=tag_list)
        wandb.define_metric("val_steps")
        wandb.define_metric("validation/*", step_metric="val_steps")
        
    dist.barrier()

    # build model
    #model = build_distill_net(args)
    model = DisNet(args)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    # logger.info(model)
    logger.info(args)

    # build optimizer & lr scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                             T_0=args.epochs, 
                             eta_min=args.min_lr)
    scaler = amp.GradScaler()
    # scaler = None
    # scheduler = None
    # args.index_split = 0
    
    logger.info("Build model")
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=False)
    # build dataset
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int(
        (args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)

    logger.info("Initialize dataset")
    train_data, val_data, collate_fn = build_dataset(args)

    # build dataloader
    logger.info("Build dataloader")
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=False,
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True,
                                   collate_fn=train_data.collate_fn)
    if args.evaluate:
        val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
        val_loader = data.DataLoader(val_data,
                                    batch_size=args.batch_size_val,
                                    shuffle=False,
                                    num_workers=args.workers_val,
                                    pin_memory=False,
                                    sampler=val_sampler,
                                    drop_last=False,
                                    collate_fn=collate_fn)
        
        # test_sampler = data.distributed.DistributedSampler(test_data, shuffle=False)
        # test_loader = data.DataLoader(test_data,
        #                             batch_size=args.batch_size_val,
        #                             shuffle=False,
        #                             num_workers=args.workers_val,
        #                             pin_memory=False,
        #                             sampler=test_sampler,
        #                             drop_last=False,
        #                             collate_fn=test_data.collate_fn)

    logger.info("Check resume point")
    min_sim_loss = 9999
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            checkpoint = torch.load(
                args.resume, map_location=map_location)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            val_sim_loss = checkpoint['val_sim_loss']
            min_sim_loss = checkpoint['min_sim_loss']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            
            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    logger.info("Start training")
    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)
        # train
        logger.info(f"Train epoch: {epoch_log}/{args.epochs}")
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log,  args)

        if epoch_log % args.eval_freq == 0:
            if args.evaluate:
                val_sampler.set_epoch(epoch_log)
                if args.eval_task in ["all", "segmentation"]:
                    logger.info(f"Evaluate Segmentation epoch: {epoch_log}/{args.epochs}")
                    metrics = validate_segmentation(val_loader, model, epoch_log,  args)
                
                if args.eval_task in ["all", "grounding"]:
                    logger.info(f"Evaluate Grounding epoch: {epoch_log}/{args.epochs}")
                    metrics = validate_grounding(val_loader, model, epoch_log,  args)

                if args.use_aux_loss:
                    val_sim_loss = metrics['TotalLoss']
                else:
                    val_sim_loss = metrics['DistilLoss']
            else:
                val_sim_loss = 0.0

            if args.visualize:
                logger.info("Visualization")
                visualization(test_data, model, epoch_log, args)

        dist.barrier()

        # save model
        if dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'val_sim_loss': val_sim_loss,
                    'min_sim_loss': min_sim_loss,
                }, lastname)
            
            if val_sim_loss < min_sim_loss:
                min_sim_loss = val_sim_loss
                bestname = os.path.join(args.output_dir, "best_sim_loss_model.pth")
                shutil.copyfile(lastname, bestname)
            
        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)
    if dist.get_rank() == 0:
        wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)