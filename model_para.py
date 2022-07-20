import os

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model.AAformer import CyCTR
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, step_learning_rate, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='/data/wy/few-shot-seg/cyc/cycTR_v16/exp/pascal/split0_resnet50/pascal_split0_resnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    assert args.classes > 1
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    ### multi-processing training is deprecated
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False  # sync_bn is deprecated
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    model = CyCTR(layers=args.layers, shot=args.shot, \
                  reduce_dim=args.hidden_dims, with_transformer=args.with_transformer)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "transformer" not in n
                    and p.requires_grad]},
    ]
    transformer_param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if
                       "transformer" in n and "bias" not in n and p.requires_grad],
            "lr": 1e-4,
            "weight_decay": 1e-2,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       "transformer" in n and "bias" in n and p.requires_grad],
            "lr": 1e-4,
            "weight_decay": 0,
        }
    ]
    optimizer = torch.optim.SGD(
        param_dicts,
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    transformer_optimizer = torch.optim.AdamW(transformer_param_dicts, lr=1e-4, weight_decay=1e-4)

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    # logger.info(model)
    # print(args)

    model = torch.nn.DataParallel(model.cuda())
    # 打印可训练参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("->my_loadding weight from '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            ori_state_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in ori_state_dict.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            logger.info("->loaded weight")

            for name, param in model.named_parameters(): #仅仅只包含可训练的部分
                if name in state_dict.keys():
                    param.requires_grad = False
    print("behind")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

if __name__=='__main__':
    main()