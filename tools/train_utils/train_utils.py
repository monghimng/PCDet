import torch.nn.functional as F
import torch
import os
import glob
import tqdm
from torch.nn.utils import clip_grad_norm_
import wandb
from pcdet.config import cfg
import cv2


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, tb_log=None, leave_pbar=False):
    dataloader_iter = iter(train_loader)
    total_it_each_epoch = len(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('learning_rate', cur_lr, accumulated_iter)
            tb_log.add_scalar('epoch', accumulated_iter // total_it_each_epoch, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        if cfg.INJECT_SEMANTICS:
            import numpy as np  # todo: for some reason, error is thrown if this line is at the top
            img = batch['img']
            image_shape = img.shape[2:]
            device = torch.cuda.current_device()
            img = torch.tensor(img, dtype=torch.float32, device=device)

            pred_batch = model.module.seg_model(img)
            # pred_batch = model.seg_model(img)  # if not using distrivurted

            # todo: try nearest neighbor when we pass features down because those might be more accuracte probabilities
            pred_batch = F.upsample(input=pred_batch, size=list(image_shape), mode='bilinear')

            # todo:
            # argmax strategy
            pred_batch = pred_batch.argmax(dim=1, keepdim=True)
            pred_batch = pred_batch == 13  # convert to binary mask for cars for now
            pred_batch = pred_batch.permute(0, 2, 3, 1).int().detach().cpu().numpy()
            # probability distribution strategy
            # logits strategy
            # pred_batch = pred_batch.permute(0, 2, 3, 1).detach().cpu().numpy()  # 19 class channels
            # cfg.DATA_CONFIG.NUM_POINT_FEATURES['total'] = 3 + 19
            # cfg.DATA_CONFIG.NUM_POINT_FEATURES['use'] = 3 + 19

            # project pts onto image to get the point color
            semantic_pts_lst = []
            for pts, segmentation, calib, img_true_size in zip(batch['points_unmerged'], pred_batch, batch['calib'], batch['image_shape']):

                # in kitti, each image could be of different size, we must resize segmentation to the right size
                true_h, true_w = img_true_size
                segmentation = cv2.resize(segmentation, (true_w, true_h),
                                 interpolation=cv2.INTER_NEAREST)
                segmentation = segmentation.reshape([true_h, true_w, -1])
                pts = pts[:, :3]  # only take xyz
                img_coords, _ = calib.lidar_to_img(pts)
                img_coords = img_coords.astype(np.int32)  # note that first col is the cols, second col is the rows
                rows = img_coords[:, 1]
                cols = img_coords[:, 0]
                semantics = segmentation[rows, cols]

                # todo: potential way to play with the semantics
                semantics = semantics.astype(np.float32)
                # semantics /= 255  # normalize to [0, 1]

                # append each pt with its semantics
                semantic_pts = np.hstack([pts, semantics])
                semantic_pts_lst.append(semantic_pts)
            batch['points_unmerged'] = np.array(semantic_pts_lst)

        if cfg.VOXELIZE_IN_MODEL_FORWARD:

            # things in the batches are still np, the model decorator converts them to pytorch later
            points_batch = batch['points_unmerged']
            voxel_generator = train_loader.dataset.voxel_generator

            from collections import defaultdict
            example_merged = defaultdict(list)
            for points in points_batch:
                voxel_grid = voxel_generator.generate(points)
                voxels = voxel_grid["voxels"]
                coordinates = voxel_grid["coordinates"]
                num_points = voxel_grid["num_points_per_voxel"]

                voxel_centers = (coordinates[:, ::-1] + 0.5) * voxel_generator.voxel_size \
                                + voxel_generator.point_cloud_range[0:3]
                example_merged['voxels'].append(voxels)
                example_merged['num_points'].append(num_points)
                example_merged['coordinates'].append(coordinates)
                example_merged['voxel_centers'].append(voxel_centers)

            # need to manually collate them here
            import numpy as np
            ret = {}
            for key, elems in example_merged.items():
                if key in ['voxels', 'num_points', 'voxel_centers', 'seg_labels', 'part_labels', 'bbox_reg_labels']:
                    ret[key] = np.concatenate(elems, axis=0)
                elif key in ['coordinates', 'points']:
                    coors = []
                    for i, coor in enumerate(elems):
                        # top_row, bottom_row, rightmost col are zero, while leftmost col padding of 1
                        pad_width = ((0, 0), (1, 0))
                        coor_pad = np.pad(coor, pad_width, mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = 0
                    batch_size = elems.__len__()
                    for k in range(batch_size):
                        max_gt = max(max_gt, elems[k].__len__())
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, elems[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :elems[k].__len__(), :] = elems[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(elems, axis=0)
                # check if they are the same
                # import pdb; pdb.set_trace()
                # batch['voxels'][:, :, :3] == ret['voxels'][:, :, :3]
            batch.update(ret)

        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train_loss', loss, accumulated_iter)
                tb_log.add_scalar('learning_rate', cur_lr, accumulated_iter)

                for key, val in tb_dict.items():
                    # log some bev images every epoch
                    if 'image' in key:
                        # log x times per epoch
                        if cur_it % (total_it_each_epoch // 5) == 0:
                        # if True:
                            wandb.log({key: wandb.Image(val, caption=batch['sample_idx'][0])})
                    else:
                        tb_log.add_scalar('train_' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs)
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
