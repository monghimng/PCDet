from .detector3d import Detector3D
import pcdet.experiments as exp
import torch.nn.functional as F
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
        out_channels_bev = 2  # todo
        # this was calculated by counting number of positive pixels for each cls
        # pos_weights = torch.Tensor([1.7736, 28.0409]).cuda() / 2# todo
        pos_weights = torch.Tensor([28.0409]).cuda()
        pos_weights = torch.Tensor([28.0409]).cuda() / 2
        pos_weights = torch.Tensor([2]).cuda()  # those calculated weights don't seem to work
        pos_weights = torch.Tensor([1.5]).cuda()  # those calculated weights don't seem to work
        # self.bev_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.bev_loss = nn.BCEWithLogitsLoss()
        # self.bev_loss = nn.L1Loss()
        # self.bev_loss = FocalLoss(alpha=pos_weights, logits=True)

        # num_block = 6
        # div_factor_for_channel = (in_channels_bev / out_channels_bev) ** (1 / num_block)
        # print(div_factor_for_channel)
        # blocks = []
        #
        # # todo: this is only for debug
        # lst = [1, 4, 16, 64, 384]
        # for ck, in_channels_bev in zip(lst[:-1], lst[1:]):
        #     blocks.append(nn.Conv2d(ck, in_channels_bev, 3, padding=1))
        #     blocks.append(nn.BatchNorm2d(in_channels_bev))
        #     blocks.append(nn.ReLU())
        # blocks.append((nn.Conv2d(384, 384, 3, stride=2, padding=1)))
        #
        # for j in range(num_block):
        #     in_chan = int(in_channels_bev // (div_factor_for_channel ** j))
        #     out_chan = int(in_channels_bev // (div_factor_for_channel ** (j + 1)))
        #     print(out_chan)
        #     blocks.append(nn.Conv2d(in_chan, out_chan, 1))
        #     blocks.append(nn.ReLU())
        #     blocks.append(nn.Conv2d(out_chan, out_chan, 3, padding=1))
        #     blocks.append(nn.BatchNorm2d(out_chan))
        #     blocks.append(nn.ReLU())
        # blocks.append(nn.Conv2d(out_channels_bev, out_channels_bev, 3, padding=1, bias=True))

        import segmentation_models_pytorch as smp
        self.bev_conv = smp.Unet('resnet18', encoder_weights='imagenet', classes=out_channels_bev)
        self.bev_conv.encoder.conv1 = nn.Conv2d(in_channels_bev, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.bev_conv = nn.Sequential(*blocks)

        # freeze model to debug
        # for param in self.vfe.parameters():
        #     param.requires_grad = False
        # for param in self.rpn_net.parameters():
        #     param.requires_grad = False
        # for param in self.rpn_head.parameters():
        #     param.requires_grad = False



    def forward_rpn(self, voxels, num_points, coordinates, batch_size, voxel_centers, **kwargs):

        # v = voxels.detach()
        # v = v.reshape([v.shape[0], -1])
        # self.rpn_net.nchannels = 128
        # vv = self.rpn_net(v, coordinates, batch_size, output_shape=self.grid_size[::-1])
        # self.rpn_net.nchannels = 64
        # vv = vv.reshape([vv.shape[0], 32, 4, vv.shape[-2], vv.shape[-1]])
        # tag_only = vv[:, :, 3, :, :]
        # aggregated = tag_only.max(dim=1)[0]
        # aggregated = torch.unsqueeze(aggregated, dim=1)



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
            'spatial_features_last': rpn_preds_dict['spatial_features_last'],
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
        # import pdb; pdb.set_trace()
        rpn_ret_dict = self.forward_rpn(**input_dict)

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()

            bev_loss, bev_tb_dict = exp.after_stage1_rpn_net(self, rpn_ret_dict, input_dict)
            tb_dict.update(bev_tb_dict)
            loss = 0.0000001 * loss + bev_loss
            ret_dict = {
                'loss': loss,
            }

            return ret_dict, tb_dict, disp_dict
        else:
            exp.after_stage1_rpn_net(self, rpn_ret_dict, input_dict)
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
