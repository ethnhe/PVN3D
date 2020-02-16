from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.utils.meanshift_pytorch import MeanShiftTorch


class FocalLoss(_Loss):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def of_l1_loss(
        pred_ofsts, kp_targ_ofst, labels,
        sigma=1.0, normalize=True, reduce=False
):
    '''
    :param pred_ofsts:      [bs, n_kpts, n_pts, c]
    :param kp_targ_ofst:    [bs, n_pts, n_kpts, c]
    :param labels:          [bs, n_pts, 1]
    '''
    w = (labels > 1e-8).float()
    bs, n_kpts, n_pts, c = pred_ofsts.size()
    sigma_2 = sigma ** 3
    w = w.view(bs, 1, n_pts, 1).repeat(1, n_kpts, 1, 1).contiguous()
    kp_targ_ofst = kp_targ_ofst.view(bs, n_pts, n_kpts, 3)
    kp_targ_ofst = kp_targ_ofst.permute(0, 2, 1, 3).contiguous()
    diff = pred_ofsts - kp_targ_ofst
    abs_diff = torch.abs(diff)
    abs_diff = w * abs_diff
    in_loss = abs_diff

    if normalize:
        in_loss = torch.sum(
            in_loss.view(bs, n_kpts, -1), 2
        ) / (torch.sum(w.view(bs, n_kpts, -1), 2) + 1e-3)

    if reduce:
        torch.mean(in_loss)

    return in_loss


class OFLoss(_Loss):
    def __init__(self):
        super(OFLoss, self).__init__(True)

    def forward(
            self, pred_ofsts, kp_targ_ofst, labels,
            normalize=True, reduce=False
    ):
        l1_loss = of_l1_loss(
            pred_ofsts, kp_targ_ofst, labels,
            sigma=1.0, normalize=True, reduce=False
        )

        return l1_loss

