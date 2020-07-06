from .detector3d import Detector3D
from ...config import cfg
import numpy as np
import torch
import torch.nn as nn
from ...utils.metrics import Evaluator

class PointPillar(Detector3D):
    """
    PointPillar network from https://arxiv.org/abs/1812.05784. This is a 1 stage detector consisted of
        vfe: PillarFeatureNetOld2. See vfe_utils.py for details.
        rpn: PointPillarsScatter. See pillar_scatter.py for details.
        rpn_head: RPNV2.

    """
    def __init__(self, num_class, dataset):
        super().__init__(num_class, dataset)
        self.build_networks(cfg.MODEL)

        # bev conv
        in_channels_bev = 384
        out_channels_bev = 2
        self.bev_conv = nn.Conv2d(in_channels_bev, out_channels_bev, 3, padding=1, bias=True)
        self.bev_loss = nn.BCEWithLogitsLoss()

    def forward_rpn(self, voxels, num_points, coordinates, batch_size, voxel_centers, **kwargs):
        voxel_features = self.vfe(
            features=voxels,
            num_voxels=num_points,
            coords=coordinates
        )
        spatial_features = self.rpn_net(
            voxel_features, coordinates, batch_size,
            output_shape=self.grid_size[::-1]
        )
        rpn_preds_dict = self.rpn_head(
            spatial_features,
            **{'gt_boxes': kwargs.get('gt_boxes', None)}
        )

        rpn_ret_dict = {
            'rpn_cls_preds': rpn_preds_dict['cls_preds'],
            'rpn_box_preds': rpn_preds_dict['box_preds'],
            'rpn_dir_cls_preds': rpn_preds_dict.get('dir_cls_preds', None),
            'anchors': rpn_preds_dict['anchors'],
            'spatial_features_last': rpn_preds_dict['spatial_features_last']
        }
        return rpn_ret_dict

    # def forward(self, input_dict):
    #     rpn_ret_dict = self.forward_rpn(**input_dict)
    #
    #     if self.training:
    #         loss, tb_dict, disp_dict = self.get_training_loss()
    #
    #         ret_dict = {
    #             'loss': loss
    #         }
    #         return ret_dict, tb_dict, disp_dict
    #     else:
    #         pred_dicts, recall_dicts = self.predict_boxes(rpn_ret_dict, rcnn_ret_dict=None, input_dict=input_dict)
    #         return pred_dicts, recall_dicts
    #
    # def get_training_loss(self):
    #     disp_dict = {}
    #
    #     loss_anchor_box, tb_dict = self.rpn_head.get_loss()
    #     loss_rpn = loss_anchor_box
    #     tb_dict = {
    #         'loss_rpn': loss_rpn.item(),
    #         **tb_dict
    #     }
    #
    #     loss = loss_rpn
    #     return loss, tb_dict, disp_dict

    def forward(self, input_dict):
        rpn_ret_dict = self.forward_rpn(**input_dict)

        # obtain bev features in the right output dimension
        rpn_features = rpn_ret_dict['spatial_features_last']
        bev_features = self.bev_conv(rpn_features)
        gt = input_dict['bev'].astype(np.int32)

        # compute predictions for both training and validation
        logits = bev_features.detach().cpu().numpy()
        predictions = logits > 0
        predictions = predictions.astype(np.int32)
        evaluator = Evaluator(2)
        evaluator.add_batch(gt, predictions)
        iou = evaluator.Pixel_Accuracy()

        if self.training:

            # compute loss
            gt_tensor = torch.tensor(gt, dtype=torch.float32, device=torch.cuda.current_device())
            bev_loss = self.bev_loss(bev_features, gt_tensor)

            loss, tb_dict, disp_dict = self.get_training_loss()
            loss = 0.01 + bev_loss
            tb_dict['iou'] = iou
            tb_dict['bev_loss'] = bev_loss.item()

            ret_dict = {
                'loss': loss,
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.predict_boxes(rpn_ret_dict, rcnn_ret_dict=None, input_dict=input_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_anchor_box, tb_dict = self.rpn_head.get_loss()
        loss_rpn = loss_anchor_box
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
