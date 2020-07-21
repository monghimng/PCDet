import tqdm
import cv2
import torch.nn.functional as F
import torch
import time
import pickle
from pcdet.config import cfg
from pcdet.models import example_convert_to_torch


def statistics_info(ret_dict, metric, disp_dict):
    if cfg.MODEL.RCNN.ENABLED:
        for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
            metric['recall_roi_%s' % str(cur_thresh)] += ret_dict['roi_%s' % str(cur_thresh)]
            metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict['rcnn_%s' % str(cur_thresh)]
        metric['gt_num'] += ret_dict['gt']
        min_thresh = cfg.MODEL.TEST.RECALL_THRESH_LIST[0]
        disp_dict['recall_%s' % str(min_thresh)] = \
            '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(model, dataloader, epoch_id, logger, save_to_file=False, result_dir=None, test_mode=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    if save_to_file:
        final_output_dir = result_dir / 'final_result' / 'data'
        final_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        final_output_dir = None

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    model.eval()

    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, data in enumerate(dataloader):

        ############################## for PSF and voxelization in forward pass
        batch = data
        if cfg.INJECT_SEMANTICS:
            import numpy as np  # todo: for some reason, error is thrown if this line is at the top
            img = batch['img']
            image_shape = img.shape[2:]
            device = torch.cuda.current_device()
            img = torch.tensor(img, dtype=torch.float32, device=device)

            pred_batch = model.seg_model(img)
            # pred_batch = model.seg_model(img)  # if not using distrivurted

            # todo: try nearest neighbor when we pass features down because those might be more accuracte probabilities
            pred_batch = F.upsample(input=pred_batch, size=list(image_shape), mode='bilinear')

            if cfg.INJECT_SEMANTICS_MODE == 'binary_car_mask':
                # argmax strategy
                pred_batch = pred_batch.argmax(dim=1, keepdim=True)
                pred_batch = pred_batch == 13  # convert to binary mask for cars for now
                pred_batch = pred_batch.int()
                pred_batch = pred_batch.permute(0, 2, 3, 1).detach().cpu().numpy()
            elif cfg.INJECT_SEMANTICS_MODE == 'logit_car_mask':
                pred_batch = pred_batch[:, 13: 14, :, :]
                pred_batch = pred_batch.permute(0, 2, 3, 1).detach().cpu().numpy()
                
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
            voxel_generator = dataloader.dataset.voxel_generator

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

        ############################## end for PSF and voxelization in forward pass
        input_dict = example_convert_to_torch(data)
        pred_dicts, ret_dict = model(input_dict)
        disp_dict = {}

        statistics_info(ret_dict, metric, disp_dict)
        annos = dataset.generate_annotations(input_dict, pred_dicts, class_names,
                                             save_to_file=save_to_file, output_dir=final_output_dir)
        det_annos += annos
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

    progress_bar.close()

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    ret_dict = {}
    if cfg.MODEL.RCNN.ENABLED:
        gt_num_cnt = metric['gt_num']
        for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall_roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall_rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['num_example']
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(det_annos, class_names, eval_metric=cfg.MODEL.TEST.EVAL_METRIC)

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
