from .detector3d import Detector3D
import wandb
from ...config import cfg
import numpy as np
import torch
import torch.nn as nn
from ...utils.metrics import Evaluator

# for calculating weights in bce losses
# pos_samples = 0
# neg_samples = 0

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
        # this was calculated by counting number of positive pixels for each cls
        pos_weights = torch.Tensor([1.7736, 28.0409]).cuda() / 2
        self.bev_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

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

        if self.training:

            ############################## compute loss #############################
            num_classes = 2

            # obtain bev features in the right output dimension
            rpn_features = rpn_ret_dict['spatial_features_last']
            bev_features = self.bev_conv(rpn_features)
            gt = input_dict['bev'].astype(np.int32)

            # torch loss only takes in tensor :(
            gt_tensor = torch.tensor(gt, dtype=torch.float32, device=torch.cuda.current_device())

            # reshape both bev_features and gt to [N * H * W, num_classes] because bceloss treats
            # the second dimension as different classes and compute them separately
            # since the doc was written for linear layers, not sure what happened if H and W
            # dimensions are also provided. Doing this just in case.
            bev_features_reshaped = bev_features.permute(0, 2, 3, 1).reshape([-1, num_classes])
            gt_tensor_reshaped = gt_tensor.permute(0, 2, 3, 1).reshape([-1, num_classes])

            # # compute positive weights for each classes to focus more on positive classes for bcelogit loss
            # # only need to do this once to obtain those weights
            # # uncomment this to get the stats
            # global pos_samples
            # global neg_samples
            # pos_samples += gt_tensor_reshaped.sum(dim=0)
            # neg_samples += len(gt_tensor_reshaped) - gt_tensor_reshaped.sum(dim=0)
            # print("pos_weights:{} total_pos_samples: {} total_neg_samples: {}".format(neg_samples / pos_samples,
            #       pos_samples, neg_samples))

            bev_loss = self.bev_loss(bev_features_reshaped, gt_tensor_reshaped)
            # bev_loss = self.bev_loss(bev_features, gt_tensor)

            loss, tb_dict, disp_dict = self.get_training_loss()
            loss = 0.01 * loss + bev_loss
            tb_dict['bev_loss'] = bev_loss.item()

            ret_dict = {
                'loss': loss,
            }

            ############################## somtimes log bev to wandb ##############################

            # make predictions based on the logits
            logits = bev_features.detach().cpu().numpy()
            predictions = logits > 0
            predictions = predictions.astype(np.int32)

            for cls_idx in range(num_classes):
                # pick the 0th image in this sample and log by different cls
                tb_dict['image_bev_predicted_cls{}'.format(cls_idx + 1)] = predictions[0, cls_idx]
                tb_dict['image_bev_gt_cls{}'.format(cls_idx + 1)] = gt[0, cls_idx]

            ############################## compute iou metrics ##############################

            # construct an evaluator to compute iou for this batch to give intuition in training
            evaluator = Evaluator(
                1 + num_classes)  # plus one for the void region not occupied by anything and also nondrivable

            # make copies because we don't want fns that depend on gt & predictions to change
            gt = np.array(gt)
            predictions = np.array(predictions)

            # multiply by the channel num to differentiate between classes
            for cls_idx in range(num_classes):
                gt[:, cls_idx, :, :] *= cls_idx + 1  # plus 1 because want to leave 0 to represent void
                predictions[:, cls_idx, :, :] *= cls_idx + 1

            evaluator.add_batch(gt, predictions)
            ciou = evaluator.class_iou()
            for cls_idx in range(num_classes):
                tb_dict['iou_cls{}'.format(cls_idx + 1)] = ciou[cls_idx + 1]
            miou = ciou[1:].mean()  # compute miou without the void class
            tb_dict['miou'] = miou

            # debugging code to verify if the way I compute ciou is correct
            # i = 0
            # i = 1
            # evaluator = Evaluator(2)
            # evaluator.add_batch(gt[:, i], predictions[:, i])
            # ciou = evaluator.class_iou()
            # print(ciou)

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

