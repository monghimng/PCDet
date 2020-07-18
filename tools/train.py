import wandb
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from pcdet.config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import torch.distributed as dist

from pathlib import Path
import argparse
import datetime
import glob


def parge_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=80, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=10000, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parge_config()
    if args.launcher == 'none':
        dist_train = False
    else:
        args.batch_size, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.batch_size, args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        total_gpus = dist.get_world_size()
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    if args.local_rank == 0:
        wandb.init(project='BEVSEG-PCDet', sync_tensorboard=True, name=args.extra_tag, config={**vars(args), **cfg})

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        cfg.DATA_CONFIG.DATA_DIR, args.batch_size, dist_train, workers=args.workers, logger=logger, training=True
    )

    model = build_network(train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.MODEL.TRAIN.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()],
            # find_unused_parameters=True # uncomment this line to debug unused params
        )
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.MODEL.TRAIN.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s(%s)**********************' % (cfg.TAG, args.extra_tag))
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.MODEL.TRAIN.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num
    )

    logger.info('**********************End training**********************')

    '''
    :param images: set of images (rgb images, other form of inputs) to_be_bilinearly_interpolated
    :param labels: set of images (segmentation maps, depth maps) to be interpolated with nearest neighbor
    '''
import cv2
import numpy as np
def image_resize(image, long_size, label=None):
    h, w = image.shape[:2]
    if h > w:
        new_h = long_size
        new_w = np.int(w * long_size / h + 0.5)
    else:
        new_w = long_size
        new_h = np.int(h * long_size / w + 0.5)

    image = cv2.resize(image, (new_w, new_h),
                       interpolation=cv2.INTER_LINEAR)
    if label is not None:
        label = cv2.resize(label, (new_w, new_h),
                           interpolation=cv2.INTER_NEAREST)
    else:
        return image

    return image, label
def input_transform(image, mean, std):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image
if __name__ == '__main__':
    # from hrnet.tools.train import parse_args_and_construct_model
    # seg_args = ''' --cfg /home/eecs/monghim.ng/BESEG/hrnet/experiments/cityscapes/cityscapes_pcdet.yaml \
    # MODEL.PRETRAINED /home/eecs/monghim.ng/BESEG/hrnet/hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
    # '''
    # seg_model = parse_args_and_construct_model(seg_args)
    #
    # img_path = '/data/ck/data/argoverse/argoverse-tracking-kitti-format/training/image_2/000000000.png'
    # img_path = '/data/ck/data/argoverse/argoverse-tracking-kitti-format/training/image_2/000000000.png'
    # img_path = '/home/eecs/monghim.ng/transfer/aachen_000000_000019_leftImg8bit.png'
    # img_path = '/data/ck/data/kitti_obj_det/training/image_2/000000.png'
    # img_path = '/data/ck/data/kitti_obj_det/training/image_2/001000.png'
    # # img_path = '/data/ck/data/kitti_obj_det/training/image_2/002000.png'
    # # img_path = '/data/ck/data/kitti_obj_det/training/image_2/003000.png'
    # img = cv2.imread(img_path)
    #
    # # cityscapes image preprocessing: resize, normalize
    # long_size = 2048
    # # long_size = 1536
    # img = image_resize(img, long_size)
    # shape = img.shape
    #
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # img = input_transform(img, mean, std)
    # img = img.transpose((2, 0, 1))
    #
    # batch = [img]
    # batch_tensor = torch.Tensor(batch)
    # pred_batch = seg_model(batch_tensor)
    # pred_batch = F.upsample(input=pred_batch, size=shape[:2], mode='bilinear')
    #
    # pred_batch = pred_batch.argmax(dim=1)
    # pred = pred_batch[0]
    #
    # pred = pred.detach().cpu().numpy().astype(np.int32)
    # # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_0.png', pred)
    # # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_1.png', pred)
    # # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_2.png', pred)
    # # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_3.png', pred)
    # # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_2048res_0.png', pred)
    # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_2048res_1.png', pred)
    # # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_2048res_2.png', pred)
    # # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_2048res_3.png', pred)
    # # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_1536res_2.png', pred)
    # # cv2.imwrite('/home/eecs/monghim.ng/transfer/hrnet_kitti_1536res_1.png', pred)
    # exit()

    main()
