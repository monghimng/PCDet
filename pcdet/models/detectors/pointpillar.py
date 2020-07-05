from .detector3d import Detector3D
from ...config import cfg

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

        # ckk


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
            'anchors': rpn_preds_dict['anchors']
        }
        return rpn_ret_dict

    def forward(self, input_dict):
        rpn_ret_dict = self.forward_rpn(**input_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
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

    # def forward(self, input_dict):
    #     rpn_ret_dict = self.forward_rpn(**input_dict)
    #
    #     # obtain bev features in the right output dimension
    #     rpn_features = rpn_ret_dict['spatial_features_last']
    #     bev_features = self.bev_conv(rpn_features)
    #
    #     # compute predictions for both training and validation
    #
    #     if self.training:
    #
    #         ret_dict = {
    #             'loss': loss
    #         }
    #
    #         # include miou and iou for each class in tb
    #         tb_dict = {
    #
    #         }
    #         disp_dict = None
    #         return ret_dict, tb_dict, disp_dict
    #     else:
    #         pred_dicts, recall_dicts = self.predict_boxes(rpn_ret_dict, rcnn_ret_dict=None, input_dict=input_dict)
    #         return pred_dicts, recall_dicts
