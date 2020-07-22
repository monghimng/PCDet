from .detector3d import Detector3D
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
        out_channels_bev = 2 #todo
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

            ############################## compute loss #############################
            num_classes = 2

            # obtain bev features in the right output dimension

            # opt1: using poitnpillar features
            rpn_features = rpn_ret_dict['spatial_features_last']
            # opt2: using projected points # todo this is cheating :D
            # rpn_features = rpn_ret_dict['ck']
            rpn_features = F.interpolate(rpn_features, size=416, mode='bilinear')
            # rpn_features = torch.cat([rpn_features, rpn_features, rpn_features], dim=1)  # duplicate 3 times to fit usual rgb images

            bev_features = self.bev_conv(rpn_features)
            bev_features = F.interpolate(bev_features, size=200, mode='bilinear')

            gt = input_dict['bev'].astype(np.int32)
            # gt = input_dict['bev'].astype(np.int32)[:, 1: 2]

            gt = np.transpose(gt, [0, 1, 3, 2])
            gt = gt[:, :, ::-1, ::-1]
            gt = np.ascontiguousarray(gt)

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
            loss = 0.0000001 * loss + bev_loss
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
                tb_dict['image_bev_projected_pts_cls{}'.format(cls_idx + 1)] = rpn_features[0, cls_idx]
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
