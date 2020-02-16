import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from lib.pointnet import PointNetfeat

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()

    def forward(self, x):
        x = self.model(x)
        return x

# class ModifiedResnet(nn.Module):
#
#     def __init__(self, usegpu=True):
#         super(ModifiedResnet, self).__init__()
#
#         self.model = psp_models['resnet34'.lower()]()
#         # self.model = nn.DataParallel(self.model)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
#         self.conv1 = torch.nn.Conv1d(1088, 512, 1)
#         self.conv2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x)
#         x = x.transpose(2,1).contiguous()
#         x = F.log_softmax(x.view(-1,self.k), dim=-1)
#         x = x.view(batchsize, n_pts, self.k)
#         return x, trans, trans_feat
#

class ConvFeat(nn.Module):
    def __init__(self):
        super(ConvFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(1888, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, emb):
        x = F.relu(self.bn1(self.conv1(emb)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class SGKPNet(nn.Module):
    def __init__(self, num_points, num_obj, num_kps):
        super(SGKPNet, self).__init__()
        self.num_obj = num_obj
        self.num_points = num_points

        # self.log_sigma1_square = Variable(
        #     torch.from_numpy(np.array([-2]).astype(np.float32))
        # ).cuda()
        # self.log_sigma2_square = Variable(
        #     torch.from_numpy(np.array([4]).astype(np.float32))
        # ).cuda()
        # self.sigma1_square = torch.exp(self.log_sigma1_square)
        # self.sigma2_square = torch.exp(self.log_sigma2_square)

        self.num_kps = num_kps
        self.cnn = ModifiedResnet()
        self.pntf = PointNetfeat(global_feat=False, feature_transform=False)
        self.rgb_emb_conv1 = torch.nn.Conv1d(32, 256, 1)
        self.rgb_emb_conv2 = torch.nn.Conv1d(256, 512, 1)

        self.rgb_emb_bn1 = nn.BatchNorm1d(256)
        self.rgb_emb_bn2 = nn.BatchNorm1d(512)

        self.cf_seg = ConvFeat()
        self.cf_kp = ConvFeat()
        self.cf_c = ConvFeat()

        self.conv_seg = torch.nn.Conv1d(128, self.num_obj, 1)
        self.conv_kp = torch.nn.Conv1d(128, num_kps*3, 1) # kpts
        self.conv_nm = torch.nn.Conv1d(128, 3, 1) # normal
        self.conv_c = torch.nn.Conv1d(128, num_kps*1, 1) # confidence

    def forward(self, img, x, choose):
        out_img = self.cnn(img)
        bs, di, _, _ = out_img.size()
        rgb_emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        rgb_emb1 = torch.gather(rgb_emb, 2, choose).contiguous()
        rgb_emb2 = F.relu(self.rgb_emb_bn1(self.rgb_emb_conv1(rgb_emb1)))
        rgb_emb3 = F.relu(self.rgb_emb_bn2(self.rgb_emb_conv2(rgb_emb2)))
        rgb_emb = torch.cat([rgb_emb1, rgb_emb2, rgb_emb3], dim=1) # 32+256+512=800

        x = x.transpose(2, 1).contiguous()
        d_emb, trans, trans_feat = self.pntf(x) # 1088
        rgbd_emb = torch.cat([d_emb, rgb_emb], dim=1) # 1088 + 800 = 1888

        # seg
        n_pts = self.num_points
        x_seg = self.cf_seg(rgbd_emb)
        x_seg = self.conv_seg(x_seg)
        x_seg = x_seg.transpose(2,1).contiguous() # [bs, nobj, npts] to [bs, npts, nobj]
        x_seg = F.log_softmax(x_seg.view(-1,self.num_obj), dim=-1)
        x_seg = x_seg.view(bs, n_pts, self.num_obj)

        # keypoint
        x_kp = self.cf_kp(rgbd_emb)
        x_c = self.cf_kp(rgbd_emb)

        x_kp = self.conv_kp(x_kp).view(bs, self.num_kps, 3, self.num_points)
        x_kp = x_kp.permute(0, 1, 3, 2).contiguous()
        x_c = torch.sigmoid(self.conv_c(x_c)).view(bs, self.num_kps, 1, self.num_points)
        x_c = x_c.permute(0, 1, 3, 2).contiguous()

        return x_seg, x_kp, x_c, trans, trans_feat #, self.sigma1_square, self.sigma2_square, self.log_sigma1_square, self.log_sigma2_square


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.e_bn1 = nn.BatchNorm1d(64)
        # self.e_bn2 = nn.BatchNorm1d(128)
        # self.bn5 = nn.BatchNorm1d(512)
        # self.bn6 = nn.BatchNorm1d(1024)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class KPNet(nn.Module):
    def __init__(self, num_points, num_obj, num_kps):
        super(KPNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)

        self.conv1_kp = torch.nn.Conv1d(1408, 1024, 1)
        self.conv1_nm = torch.nn.Conv1d(1408, 1024, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 1024, 1)

        self.conv2_kp = torch.nn.Conv1d(1024, 512, 1)
        self.conv2_nm = torch.nn.Conv1d(1024, 512, 1)
        self.conv2_c = torch.nn.Conv1d(1024, 512, 1)

        self.conv3_kp = torch.nn.Conv1d(512, num_obj*num_kps*3, 1) # kpts
        self.conv3_nm = torch.nn.Conv1d(512, 3, 1) # normal
        self.conv3_c = torch.nn.Conv1d(512, num_obj*num_kps*1, 1) # confidence

        # self.bn1_kp = nn.BatchNorm1d(1024)
        # self.bn2_kp = nn.BatchNorm1d(512)

        # self.bn1_c = nn.BatchNorm1d(1024)
        # self.bn2_c = nn.BatchNorm1d(512)

        self.num_obj = num_obj
        self.num_kps = num_kps

    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        kpx = F.relu(self.conv1_kp(ap_x))
        nmx = F.relu(self.conv1_nm(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        kpx = F.relu(self.conv2_kp(kpx))
        nmx = F.relu(self.conv2_nm(nmx))
        cx = F.relu(self.conv2_c(cx))

        kpx = self.conv3_kp(kpx).view(bs, self.num_obj, self.num_kps, 3, self.num_points)
        nmx = self.conv3_nm(nmx).view(bs, 3, self.num_points)
        cx = torch.sigmoid(self.conv3_c(cx)).view(bs, self.num_obj, self.num_kps, 1, self.num_points)
        # cx = self.conv4_c(cx).view(bs, self.num_obj, self.num_kps, 1, self.num_points)
        for idx in range(bs):
            tmp_out_kpx = torch.index_select(kpx[idx], 0, obj[idx])
            tmp_out_cx = torch.index_select(cx[idx], 0, obj[idx])
            if idx == 0:
                out_kpx = tmp_out_kpx
                out_cx = tmp_out_cx
            else:
                out_kpx = torch.cat((out_kpx, tmp_out_kpx), 0)
                out_cx = torch.cat((out_cx, tmp_out_cx), 0)

        out_kpx = out_kpx.view(bs, self.num_kps, 3, self.num_points)
        out_nmx = nmx.view(bs, 3, self.num_points)
        out_cx = out_cx.view(bs, self.num_kps, 1, self.num_points)
        out_kpx = out_kpx.contiguous().transpose(2, 1).contiguous()
        out_nmx = out_nmx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        # out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_kpx, out_nmx, out_cx

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)

        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx, emb.detach()



class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
