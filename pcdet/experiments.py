import tqdm
from pcdet.utils import calibration
import cv2
import torch.nn.functional as F
import torch
import time
import pickle
from pcdet.config import cfg

def training_before_epoch(model):
    seg_model = None
    if cfg.INJECT_SEMANTICS:
        try:
            seg_model = model.module.seg_model
        except:
            seg_model = model.seg_model

        if not cfg.TRAIN_SEMANTIC_NETWORK:
            seg_model.eval()
            for param in seg_model.parameters():
                param.requires_grad = False

    # todo: remvoe this after debuggin
    ck = torch.nn.parameter.Parameter(torch.Tensor([1])).cuda()
    b = torch.nn.parameter.Parameter(torch.Tensor([1])).cuda()
    c = torch.nn.parameter.Parameter(torch.Tensor([1])).cuda()
    d = torch.nn.parameter.Parameter(torch.Tensor([1])).cuda()
    
    return seg_model


def between_dataloading_and_feedforward(batch, model, loader):

    device = torch.cuda.current_device()

    ############################## Begin code for torch voxelization ##############################
    if cfg.TORCH_VOXEL_GENERATOR:

        ############################## Get either lidar or pseudolidar from depth map ##############################
        points_unmerged_torch = []
        if cfg.USE_PSEUDOLIDAR:
            for pts, calib, shape in zip(batch['points_unmerged'], batch['calib'], batch['image_shape']):
                print('before cuad')
                calib = calibration.Calibration_torch(calib).cuda()
                print('after cuad')

                # todo: this code is only needed for debugging lidar points, can be removed when using pl
                pts_torch = torch.Tensor(pts).cuda()[:, :3]
                pts_img_torch, pts_depth_torch = calib.lidar_to_img(pts_torch)

                ### addition code
                depth_map = torch.zeros(tuple(shape), device=device)
                cols = pts_img_torch[:, 0]
                rows = pts_img_torch[:, 1]
                cols = cols.long()
                rows = rows.long()

                # constructed depth map
                depth_map[rows, cols] = pts_depth_torch
                # top_margin, left_margin = 480, 0  # todo: need to figure out size
                # depth_map_height, depth_map_width = 960, 600
                top_margin, left_margin = 0, 0  # todo: need to figure out size
                depth_map_height, depth_map_width = shape

                # crop
                # depth_map *= ck
                depth_map = depth_map[top_margin: top_margin + depth_map_height,
                            left_margin: left_margin + depth_map_width]
                # todo: end this code is only needed for debugging lidar points, can be removed when using pl
                valid = depth_map != 0
                valid = valid.flatten()

                import numpy as np
                row_linspace = torch.arange(top_margin, top_margin + depth_map_height)
                col_linspace = torch.arange(left_margin, left_margin + depth_map_width)
                rows, cols = torch.meshgrid(row_linspace, col_linspace)

                # todo: not needed for pl
                cols = cols.flatten()[valid].cuda()
                rows = rows.flatten()[valid].cuda()
                pts_depth_torch = depth_map.flatten()[valid]
                # todo: end not needed for pl

                pts_rect_torch = calib.img_to_rect(cols, rows, pts_depth_torch)
                pts_torch = calib.rect_to_lidar(pts_rect_torch)

                # todo: this code should not be needed when using the actual pl

                # batch['points_pl_unmerged'].append(pts_torch)
                points_unmerged_torch.append(pts_torch)
        else:
            # otherwise load lidar points
            for pts in batch['points_unmerged']:
                pts_torch = torch.Tensor(pts).cuda()
                points_unmerged_torch.append(pts_torch)
        batch['points_unmerged'] = points_unmerged_torch

        ############################## Semantic fusion ##############################
        if cfg.INJECT_SEMANTICS:
            try:
                seg_model = model.module.seg_model
            except:
                seg_model = model.seg_model
                
            import numpy as np  # todo: for some reason, error is thrown if this line is at the top
            img = batch['img']
            image_shape = img.shape[2:]
            device = torch.cuda.current_device()
            img = torch.tensor(img, dtype=torch.float32, device=device)

            pred_batch = seg_model(img)

            # todo: try nearest neighbor when we pass features down because those might be more accuracte probabilities
            pred_batch = F.upsample(input=pred_batch, size=list(image_shape), mode='bilinear')

            if cfg.INJECT_SEMANTICS_MODE == 'binary_car_mask':
                # argmax strategy
                pred_batch = pred_batch.argmax(dim=1, keepdim=True)
                pred_batch = pred_batch == 13  # convert to binary mask for cars for now
                pred_batch = pred_batch.int()
            elif cfg.INJECT_SEMANTICS_MODE == 'logit_car_mask':
                pred_batch = pred_batch[:, 13: 14, :, :]

            # probability distribution strategy
            # logits strategy
            # cfg.DATA_CONFIG.NUM_POINT_FEATURES['total'] = 3 + 19
            # cfg.DATA_CONFIG.NUM_POINT_FEATURES['use'] = 3 + 19

            # project pts onto image to get the point color
            semantic_pts_lst = []
            for pts, segmentation, calib, img_true_size in zip(batch['points_unmerged'], pred_batch, batch['calib'],
                                                               batch['image_shape']):

                calib = calibration.Calibration_torch(calib).cuda()

                # in kitti, each image could be of different size, we must resize segmentation to the
                # original size for lifting pseudolidar
                true_h, true_w = img_true_size

                # segmentation = cv2.resize(segmentation, (true_w, true_h),
                #                  interpolation=cv2.INTER_NEAREST)
                segmentation = torch.unsqueeze(segmentation, dim=0)
                segmentation = F.interpolate(segmentation, size=(true_h, true_w))
                segmentation = segmentation.reshape([true_h, true_w, -1])
                pts = pts[:, :3]  # only take xyz
                img_coords, _ = calib.lidar_to_img(pts)
                img_coords = img_coords.long()  # note that first col is the cols, second col is the rows
                rows = img_coords[:, 1]
                cols = img_coords[:, 0]
                try:
                    assert (rows >= 0).all() and (rows < len(segmentation)).all() and (cols >= 0).all() and (
                        cols <= len(segmentation[0])).all()
                except:
                    import pdb;
                    pdb.set_trace()
                semantics = segmentation[rows, cols]

                if cfg.SEMANTICS_ZERO_OUT:
                    semantics *= 0

                # append each pt with its semantics
                # pts += c
                semantic_pts = torch.cat([pts, semantics], dim=1)
                semantic_pts_lst.append(semantic_pts)
            batch['points_unmerged'] = semantic_pts_lst

        ############################## Voxelize and collate ##############################
        points_batch_torch = batch['points_unmerged']
        voxel_generator = loader.dataset.voxel_generator

        from collections import defaultdict
        example_merged = defaultdict(list)
        for points_torch in points_batch_torch:
            points_np = points_torch.detach().cpu().numpy()
            voxel_grid = voxel_generator.generate(points_np)
            coordinates = voxel_grid["coordinates"]
            num_points = voxel_grid["num_points_per_voxel"]

            # instead of performing the actual voxelization, which has to be done in c code to be faster
            # and therefore indifferentiable, we get only the indices that map a pt to its proper voxel.
            # This allows gradient to flow through voxelization.

            indices = voxel_grid['voxel_pt_indices_into_original_pt_cloud']
            voxels = points_torch[indices]
            # todo: only used to debug whehter gradients are flown back to image networks
            # voxels += b / 1000 + seg_model.ck / 1000
            # -1 in the indices means unmapped pt, thus we zero those out afterward
            voxels[indices == -1] = 0

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
            if key in ['voxels']:
                ret[key] = torch.cat(elems, dim=0)

                if not cfg.TRAIN_SEMANTIC_NETWORK:
                    ret[key] = ret[key].detach()
            elif key in ['num_points', 'voxel_centers', 'seg_labels', 'part_labels', 'bbox_reg_labels']:
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
            # batch['voxels'][:, :3] == ret['voxels'][:, :, :3]
        batch.update(ret)


from ...utils.metrics import Evaluator
import numpy as np

num_classes = 2  # todo
# construct an evaluator to accumulate iou
evaluator = Evaluator(
    1 + num_classes)  # plus one for the void region not occupied by anything and also nondrivable
def testing_after_every_iter(gt, predictions):

    ############################## compute iou metrics ##############################

    # make copies because we don't want fns that depend on gt & predictions to change
    gt = np.array(gt)
    predictions = np.array(predictions)

    # gt and preds are supposed to contain binary 0 or 1 representing whether that class is present
    # multiply by the channel num to differentiate between classes
    for cls_idx in range(num_classes):
        gt[:, cls_idx, :, :] *= cls_idx + 1  # plus 1 because want to leave 0 to represent void
        predictions[:, cls_idx, :, :] *= cls_idx + 1

    evaluator.add_batch(gt, predictions)
    ciou = evaluator.class_iou()
    tb_dict = dict()
    for cls_idx in range(num_classes):
        tb_dict['iou_cls{}'.format(cls_idx + 1)] = ciou[cls_idx + 1]
    miou = ciou[1:].mean()  # compute miou without the void class
    tb_dict['miou'] = miou
    
    return tb_dict


def testing_after_all_iter():
    pass
