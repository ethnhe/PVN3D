from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import lib_old.utils.etw_pytorch_utils as pt_utils
from collections import namedtuple
from lib_old.pspnet import PSPNet, Modified_PSPNet
# from lib_old.utils.val_net_multi_thred import *
from utils.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import torch.nn.functional as F


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

modified_psp_models = {
    'resnet18': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = modified_psp_models['resnet34'.lower()]()

    def forward(self, x):
        x, x_seg = self.model(x)
        return x, x_seg


def model_fn_decorator_vismsk_metrics(
        criterion, criterion_of, criterion_kp, vis_msk=False, cal_metrics=False,
        w=0.015
):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])
    w = w

    def model_fn(
            model, data, epoch=0, eval=False, finish=False,
            vis_msk=False, cal_metrics=False, w=0.015
    ):
        if finish:
            sv_metrics()
            return None
        if eval:
            model.eval()
        with torch.set_grad_enabled(not eval):
            needed_keys = [
                'rgb', 'pcld', 'pcld_rgb', 'choose', 'labels', 'rgb_labels',
                'kp_targ_ofst', 'RTs', 'cls_ids', 'ctr_targ_ofst',
                'kp_3ds', 'ctr_3ds'
            ]
            cu_dt = {}
            for key in needed_keys:
                cu_dt[key] = data[key].to("cuda", non_blocking=True)

            pred_info = model(
                    cu_dt['pcld_rgb'], cu_dt['rgb'], cu_dt['choose']
                )

            loss_rgbd_seg = criterion(
                pred_info['pred_rgbd_seg'].view(cu_dt['labels'].numel(), -1),
                cu_dt['labels'].view(-1)
            ).sum()
            # loss_pcld = criterion(
            #     pred_pcld_seg.view(cu_dt['labels'].numel(), -1),
            #     cu_dt['labels'].view(-1)
            # ).sum()
            # loss_rgb = criterion(
            #     pred_rgb_seg.view(cu_dt['rgb_labels'].numel(), -1),
            #     cu_dt['rgb_labels'].view(-1)
            # ).sum()
            loss_kp_of = criterion_of(
                pred_info['pred_kp_of'], cu_dt['kp_targ_ofst'], cu_dt['labels'],
                smooth=False, pcld=cu_dt['pcld'], pred_c=None,
                kp_3ds=cu_dt['kp_3ds'], w=w, ofc=False
            ).sum()
            loss_ctr_of = criterion_of(
                pred_info['pred_ctr_of'], cu_dt['ctr_targ_ofst'], cu_dt['labels'],
                smooth=False, pcld=cu_dt['pcld'], pred_c=None,
                kp_3ds=cu_dt['ctr_3ds'], w=w, ofc=False
            ).sum()
            w = [1.0, 1.0, 1.0]
            loss = loss_rgbd_seg * w[0] + loss_kp_of * w[1] + \
                   loss_ctr_of * w[2]

            _, classes_rgbd = torch.max(pred_info['pred_rgbd_seg'], -1)
            acc_rgbd = (
                classes_rgbd == cu_dt['labels']
            ).float().sum() / cu_dt['labels'].numel()
            # _, classes_pcld = torch.max(pred_pcld_seg, -1)
            # acc_pcld = (classes_pcld == cu_dt['labels']).float().sum() / cu_dt['labels'].numel()
            # _, classes_rgb = torch.max(pred_rgb_seg, -1)
            # acc_rgb = (classes_rgb == cu_dt['rgb_labels']).float().sum() / cu_dt['rgb_labels'].numel()
            # masks = torch.argmax(pred_seg, dim=2)
            if eval and cal_metrics:
                bs_cls_dictkps_lst = \
                    criterion_kp(
                        classes_rgbd, None, cu_dt['pcld_rgb'][:, :, :3], pred_of, None,
                        None, None, None, None, 'test'
                    )
            if eval and cal_metrics:
                for key in data.keys():
                    data[key] = data[key].numpy()
                eval_metrics(bs_cls_dictkps_lst, data['cls_ids'], data['RTs'])

            if eval and vis_msk:
                for key in data.keys():
                    data[key] = data[key].numpy()
                sv_msk_pcld_ctr_clus_pose(
                    data['pcld'], data['rgb'], classes_rgbd,
                    pred_info['pred_ctr_of'], data['ctr_targ_ofst'],
                    data['labels'], data['K'], data['cam_scale'], epoch,
                    data['cls_ids'], data['RTs'], pred_info['pred_kp_of'],
                    min_cnt=1, ds='linemod', cls_type=cls_type
                )
            print(
                "\n",
                "acc_rgbd", acc_rgbd.item(),
                # "acc_pcld", acc_pcld.item(),
                # "acc_rgb", acc_rgb.item(),
                "loss", loss.item(),
                "\n",
                "loss_rgbd_seg", loss_rgbd_seg.item(),
                # "loss_pcld", loss_pcld.item(),
                # "loss_rgb", loss_rgb.item(),
                "loss_kp_of", loss_kp_of.item(),
                "loss_ctr_of", loss_ctr_of.item()
            )

        return ModelReturn(
            pred_info, loss,
            {
                "acc_rgbd": acc_rgbd.item(),
                # "acc_pcld": acc_pcld.item(),
                # "acc_rgb": acc_rgb.item(),
                "loss": loss.item(),
                "loss_rgbd_seg": loss_rgbd_seg.item(),
                # "loss_pcld": loss_pcld.item(),
                # "loss_rgb": loss_rgb.item(),
                "loss_kp_of": loss_kp_of.item(),
                "loss_ctr_of": loss_ctr_of.item(),
                "loss_target": loss.item(),
            }
        )

    return model_fn

class RGBDFeat(nn.Module):
    def __init__(self, num_points):
        super(RGBDFeat, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(128, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(128, 256, 1)

        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2_rgb = nn.BatchNorm1d(256)
        # self.bn4_cld = nn.BatchNorm1d(512)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([feat_1, feat_2, ap_x], 1) # 256 + 512 + 1024 = 1792


class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(
            self, num_classes, input_channels=6, use_xyz=True,
            num_kps=8, num_points=8192
    ):
        super(Pointnet2MSG, self).__init__()

        self.num_kps = num_kps
        self.num_classes = num_classes
        self.cnn = ModifiedResnet()
        self.SA_modules = nn.ModuleList()
        self.cnn = ModifiedResnet()
        self.feat = RGBDFeat(num_points)
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=2048,
                radii=[0.0175, 0.025],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.025, 0.05],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.RGBD_FC_layer = (
            pt_utils.Seq(1792)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(num_classes, activation=None)
        )

        # self.FC_layer = (
        #     pt_utils.Seq(128)
        #     .conv1d(128, bn=True, activation=nn.ReLU())
        #     .dropout()
        #     .conv1d(num_classes, activation=None)
        # )

        self.KpOF_layer = (
            pt_utils.Seq(1792)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(256, bn=True, activation=nn.ReLU())
            .conv1d(num_kps*3, activation=None)
        )

        # self.OFC_layer = (
        #     pt_utils.Seq(1792)
        #     .conv1d(1024, bn=True, activation=nn.ReLU())
        #     .dropout()
        #     .conv1d(512, bn=True, activation=nn.ReLU())
        #     .dropout()
        #     .conv1d(num_classes*num_kps, activation=nn.Sigmoid())
        # )

        self.CtrOf_layer = (
            pt_utils.Seq(1792)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

        # self.CtrOfC_layer = (
        #     pt_utils.Seq(1792)
        #     .conv1d(1024, bn=True, activation=nn.ReLU())
        #     .conv1d(512, bn=True, activation=nn.ReLU())
        #     .conv1d(128, bn=True, activation=nn.ReLU())
        #     .conv1d(num_classes, activation=nn.Sigmoid())
        # )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, rgb, choose):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        out_rgb, rgb_seg = self.cnn(rgb)

        bs, di, _, _ = out_rgb.size()
        _, N, _ = pointcloud.size()

        rgb_emb = out_rgb.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        rgb_emb = torch.gather(rgb_emb, 2, choose).contiguous()

        xyz, features = self._break_up_pc(pointcloud)
        # features = rgb_emb

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # rgbd_feature = torch.cat([rgb_emb, l_features[0]], dim=1)
        rgbd_feature = self.feat(rgb_emb, l_features[0])
        pred_rgbd_seg = self.RGBD_FC_layer(rgbd_feature).transpose(1, 2).contiguous()
        # pred_pcld_seg = self.FC_layer(l_features[0]).transpose(1, 2).contiguous()
        pred_kp_of = self.KpOF_layer(rgbd_feature).view(
            bs, self.num_kps, 3, N
        )
        # [bs, n_kps, n_pts, c]
        pred_kp_of = pred_kp_of.permute(0, 1, 3, 2).contiguous()
        # pred_c = self.OFC_layer(rgbd_feature).view(
        #     bs, self.num_classes, self.num_kps, 1, N
        # )
        # pred_c = pred_c.permute(0, 2, 4, 1, 3).contiguous()
        pred_ctr_of = self.CtrOf_layer(rgbd_feature).view(
            bs, 1, 3, N
        )
        pred_ctr_of = pred_ctr_of.permute(0, 1, 3, 2).contiguous()
        # pred_ctr_c = self.CtrOfC_layer(rgbd_feature).view(
        #     bs, self.num_classes, 1, 1, N
        # )
        # pred_ctr_c = pred_ctr_c.permute(0, 2, 4, 1, 3).contiguous()
        pred_info = dict(
            pred_rgbd_seg = pred_rgbd_seg,
            pred_kp_of = pred_kp_of,
            pred_ctr_of = pred_ctr_of,
        )

        return pred_info


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim

    B = 2
    N = 32
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3, input_channels=3)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()

    # with use_xyz=False
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3, input_channels=3, use_xyz=False)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()
