import torch.nn as nn
import torch
from .general import bbox_iou
from .postprocess import build_targets_ota
from lib.core.evaluate import SegmentationMetric
from lib.core.twinlite_loss import TverskyLoss, FocalLossSeg

class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model, img):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """

        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model, img)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model, img):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        targets0,targets1,targets2 = targets
        device = targets0.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets0, anchors = build_targets_ota(cfg, predictions[0], targets0, model, img)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        BCEcls, BCEobj, FocalSeg, TverskyDaSeg, TverskyLlSeg = self.losses

        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in predictions[0]] 

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # Losses
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                #pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets0[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                selected_tcls = targets0[i][:, 1].long()
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), selected_tcls] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss
            


        nb, _, height, width = targets1.shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)


        drivable_are_pred=predictions[1]
        _,drivable_are_gt=torch.max(targets1, 1)

        lane_line_pred=predictions[2]
        _,lane_line_gt=torch.max(targets2, 1)

        lane_line_pred = lane_line_pred[:,:, pad_h:height-pad_h, pad_w:width-pad_w]
        lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

        drivable_are_pred = drivable_are_pred[:,:, pad_h:height-pad_h, pad_w:width-pad_w]
        drivable_are_gt = drivable_are_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

        # print(drivable_are_pred.shape, drivable_are_gt.shape,lane_line_pred.shape, lane_line_gt.shape)

        lseg_focal = FocalSeg(drivable_are_pred, drivable_are_gt) + FocalSeg(lane_line_pred, lane_line_gt)
        lseg_tvl   = TverskyDaSeg(drivable_are_pred, drivable_are_gt) + TverskyLlSeg(lane_line_pred, lane_line_gt)
        

        s = 3 / no  # output count scaling
        lcls *= cfg.LOSS.CLS_GAIN * s 
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) 
        lbox *= cfg.LOSS.BOX_GAIN * s 

        lseg = lseg_focal*cfg.LOSS.FL_GAIN + lseg_tvl*cfg.LOSS.TK_GAIN


        loss = lbox + lobj + lcls + lseg
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lbox.item(), lobj.item(), lcls.item(), lseg_focal.item(), lseg_tvl.item(), lseg.item(), loss.item())


def get_loss(cfg, device):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    # segmentation loss criteria
    FocalSeg     = FocalLossSeg(mode="multiclass", alpha=0.25)
    TverskyDaSeg = TverskyLoss(mode="multiclass", alpha=0.7, beta=0.3, gamma=4.0/3, from_logits=True)
    TverskyLlSeg = TverskyLoss(mode="multiclass", alpha=0.9, beta=0.1, gamma=4.0/3, from_logits=True)
    # Focal loss
    gamma = cfg.LOSS.FL_GAMMA  # focal loss gamma
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    loss_list = [BCEcls, BCEobj, FocalSeg, TverskyDaSeg, TverskyLlSeg]
    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss

# example
# class L1_Loss(nn.Module)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
