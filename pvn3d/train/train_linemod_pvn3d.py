from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import pprint
import os.path as osp
import os
import argparse
import time
import shutil
import tqdm
from lib.utils.etw_pytorch_utils.viz import *
from lib import PVN3D
from datasets.linemod.linemod_dataset import LM_Dataset
from lib.loss import OFLoss, FocalLoss
from common import Config
from lib.utils.sync_batchnorm import convert_model
from lib.utils.warmup_scheduler import CyclicLR
from lib.utils.pvn3d_eval_utils import TorchEval
import lib.utils.etw_pytorch_utils as pt_utils
import resource
from collections import namedtuple
import pickle as pkl


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-weight_decay",
    type=float,
    default=0,
    help="L2 regularization coeff [default: 0.0]",
)
parser.add_argument(
    "-lr", type=float, default=1e-2, help="Initial learning rate [default: 1e-2]"
)
parser.add_argument(
    "-lr_decay",
    type=float,
    default=0.5,
    help="Learning rate decay gamma [default: 0.5]",
)
parser.add_argument(
    "-decay_step",
    type=float,
    default=2e5,
    help="Learning rate decay step [default: 20]",
)
parser.add_argument(
    "-bn_momentum",
    type=float,
    default=0.9,
    help="Initial batch norm momentum [default: 0.9]",
)
parser.add_argument(
    "-bn_decay",
    type=float,
    default=0.5,
    help="Batch norm momentum decay gamma [default: 0.5]",
)
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to start from"
)
parser.add_argument(
    "-epochs", type=int, default=1000, help="Number of epochs to train for"
)
parser.add_argument(
    "-run_name",
    type=str,
    default="sem_seg_run_1",
    help="Name for run in tensorboard_logger",
)
parser.add_argument(
    "-eval_net",
    action='store_true',
    help="whether is to eval net."
)
parser.add_argument(
    "--cls",
    type=str,
    default="ape",
    help="Target object. (ape, benchvise, cam, can, cat, driller," +
    "duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)
parser.add_argument(
    "--test_occ",
    action='store_true',
    help="To eval occlusion linemod or not."
)

parser.add_argument("--test", action="store_true")
parser.add_argument("--cal_metrics", action="store_true")
args = parser.parse_args()

config = Config(dataset_name='linemod', cls_type=args.cls)
lr_clip = 1e-5
bnm_clip = 1e-2


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }


def save_checkpoint(
        state, is_best, filename="checkpoint", bestname="model_best",
        bestname_pure='pvn3d_best'
):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "{}.pth.tar".format(bestname))
        shutil.copyfile(filename, "{}.pth.tar".format(bestname_pure))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        try:
            checkpoint = torch.load(filename)
        except:
            checkpoint = pkl.load(open(filename, "rb"))
        epoch = checkpoint["epoch"]
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def model_fn_decorator(
    criterion, criterion_of, test=False
):
    modelreturn = namedtuple("modelreturn", ["preds", "loss", "acc"])
    teval = TorchEval()

    def model_fn(
        model, data, epoch=0, is_eval=False, is_test=False, finish_test=False,
        obj_id=-1
    ):
        if finish_test:
            teval.cal_lm_add(obj_id, test_occ=args.test_occ)
            return None
        if is_eval:
            model.eval()
        with torch.set_grad_enabled(not is_eval):
            cu_dt = [item.to("cuda", non_blocking=True) for item in data]
            rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, ctr_targ_ofst, \
                cls_ids, rts, labels, kp_3ds, ctr_3ds = cu_dt

            pred_kp_of, pred_rgbd_seg, pred_ctr_of = model(
                cld_rgb_nrm, rgb, choose
            )

            loss_rgbd_seg = criterion(
                pred_rgbd_seg.view(labels.numel(), -1),
                labels.view(-1)
            ).sum()
            loss_kp_of = criterion_of(
                pred_kp_of, kp_targ_ofst, labels,
            ).sum()
            loss_ctr_of = criterion_of(
                pred_ctr_of, ctr_targ_ofst, labels,
            ).sum()
            w = [2.0, 1.0, 1.0]
            loss = loss_rgbd_seg * w[0] + loss_kp_of * w[1] + \
                   loss_ctr_of * w[2]

            _, classes_rgbd = torch.max(pred_rgbd_seg, -1)
            acc_rgbd = (
                classes_rgbd == labels
            ).float().sum() / labels.numel()

            if is_test:
                teval.eval_pose_parallel(
                    pcld, rgb, classes_rgbd, pred_ctr_of, ctr_targ_ofst,
                    labels, epoch, cls_ids, rts, pred_kp_of,
                    min_cnt=1, use_p2d=False, use_ctr_clus_flter=False,
                    use_ctr=True, ds_type="linemod", obj_id=obj_id
                )

        return modelreturn(
            (pred_kp_of, pred_rgbd_seg, pred_ctr_of), loss,
            {
                "acc_rgbd": acc_rgbd.item(),
                "loss": loss.item(),
                "loss_rgbd_seg": loss_rgbd_seg.item(),
                "loss_kp_of": loss_kp_of.item(),
                "loss_ctr_of": loss_ctr_of.item(),
                "loss_target": loss.item(),
            }
        )

    return model_fn


class Trainer(object):
    r"""
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    """

    def __init__(
        self,
        model,
        model_fn,
        optimizer,
        checkpoint_name="ckpt",
        best_name="best",
        lr_scheduler=None,
        bnm_scheduler=None,
        viz=None,
    ):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model,
            model_fn,
            optimizer,
            lr_scheduler,
            bnm_scheduler,
        )

        self.checkpoint_name, self.best_name = checkpoint_name, best_name

        self.training_best, self.eval_best = {}, {}
        self.viz = viz

    def eval_epoch(self, d_loader, is_test=False, obj_id=0):
        self.model.eval()

        eval_dict = {}
        total_loss = 0.0
        count = 1.0
        for i, data in tqdm.tqdm(
            enumerate(d_loader), leave=False, desc="val"
        ):
            self.optimizer.zero_grad()

            _, loss, eval_res = self.model_fn(
                self.model, data, is_eval=True, is_test=is_test,
                obj_id=obj_id
            )

            if 'loss_target' in eval_res.keys():
                total_loss += eval_res['loss_target']
            else:
                total_loss += loss.item()

            count += 1
            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]
        if is_test:
            self.model_fn(
                self.model, data, is_eval=True, is_test=is_test, finish_test=True,
                obj_id=obj_id
            )

        return total_loss / count, eval_dict

    def train(
        self,
        start_it,
        start_epoch,
        n_epochs,
        train_loader,
        test_loader=None,
        best_loss=1e4,
        log_epoch_f = None
    ):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """

        def is_to_eval(epoch, it):
            if it < 300 * 100:
                eval_frequency = (50 * 100)
            elif it < 400 * 100:
                eval_frequency = (20 * 100)
            elif it < 500 * 100:
                eval_frequency = (12 * 100)
            elif it < 600 * 100:
                eval_frequency = (8 * 100)
            elif it < 800 * 100:
                eval_frequency = (4 * 100)
            else:
                eval_frequency = (2 * 100)
            to_eval = (it % eval_frequency) == 0
            return to_eval, eval_frequency

        it = start_it
        _, eval_frequency = is_to_eval(0, it)

        with tqdm.trange(start_epoch, n_epochs + 1, desc="epochs") as tbar, tqdm.tqdm(
            total=eval_frequency, leave=False, desc="train"
        ) as pbar:

            for epoch in tbar:
                # Reset numpy seed.
                # REF: https://github.com/pytorch/pytorch/issues/5059
                np.random.seed()
                if log_epoch_f is not None:
                    os.system("echo {} > {}".format(epoch, log_epoch_f))
                for ibs, batch in enumerate(train_loader):
                    self.model.train()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step(it)

                    if self.bnm_scheduler is not None:
                        self.bnm_scheduler.step(it)

                    self.optimizer.zero_grad()
                    _, loss, res = self.model_fn(self.model, batch)

                    loss.backward()
                    self.optimizer.step()

                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()

                    if self.viz is not None:
                        self.viz.update("train", it, res)

                    eval_flag, eval_frequency = is_to_eval(epoch, it)
                    if eval_flag:
                        pbar.close()

                        if test_loader is not None:
                            val_loss, res = self.eval_epoch(test_loader)

                            if self.viz is not None:
                                self.viz.update("val", it, res)

                            is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            save_checkpoint(
                                checkpoint_state(
                                    self.model, self.optimizer, val_loss, epoch, it
                                ),
                                is_best,
                                filename=self.checkpoint_name,
                                bestname=self.best_name +'_%.4f'% val_loss,
                                bestname_pure=self.best_name,
                            )
                            info_p = self.checkpoint_name.replace(
                                '.pth.tar','_epoch.txt'
                            )
                            os.system(
                                'echo {} {} >> {}'.format(
                                    it, val_loss, info_p
                                )
                            )

                        pbar = tqdm.tqdm(
                            total=eval_frequency, leave=False, desc="train"
                        )
                        pbar.set_postfix(dict(total_it=it))

                    self.viz.flush()

        return best_loss


if __name__ == "__main__":
    print("cls_type: ", args.cls)
    if not args.eval_net:
        train_ds = LM_Dataset('train', cls_type=args.cls)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.mini_batch_size, shuffle=True,
            num_workers=20, worker_init_fn=worker_init_fn
        )
        val_ds = LM_Dataset('val', cls_type=args.cls)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.val_mini_batch_size, shuffle=False,
            num_workers=10
        )
    else:
        if args.test_occ:
            test_ds = OCC_LM_Dataset('test', cls_type=args.cls)
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
                num_workers=10
            )
        else:
            test_ds = LM_Dataset('test', cls_type=args.cls)
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
                num_workers=10
            )

    model = PVN3D(
        num_classes=config.n_classes, pcld_input_channels=6, pcld_use_xyz=True,
        num_points=config.n_sample_points
    ).cuda()
    model = convert_model(model)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    cur_mdl_pth = os.path.join(config.log_model_dir, 'pvn3d.pth.tar')
    if args.checkpoint is None and os.path.exists(cur_mdl_pth):
        args.checkpoint = cur_mdl_pth
    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = load_checkpoint(
            model, optimizer, filename=args.checkpoint
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    model = nn.DataParallel(
        model
    )

    lr_scheduler = CyclicLR(
        optimizer, base_lr=1e-5, max_lr=1e-3,
        step_size=config.n_total_epoch * config.num_mini_batch_per_epoch // 6,
        mode = 'triangular'
    )

    bnm_lmbd = lambda it: max(
        args.bn_momentum
        * args.bn_decay ** (int(it * config.mini_batch_size / args.decay_step)),
        bnm_clip,
    )
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bnm_lmbd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    model_fn = model_fn_decorator(
        nn.DataParallel(FocalLoss(gamma=2)),
        nn.DataParallel(OFLoss()),
        args.test,
    )

    viz = CmdLineViz()

    viz.text(pprint.pformat(vars(args)))

    checkpoint_fd = config.log_model_dir

    trainer = Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name = os.path.join(checkpoint_fd, "{}_pvn3d".format(args.cls)),
        best_name = os.path.join(checkpoint_fd, "{}_pvn3d_best".format(args.cls)),
        lr_scheduler = lr_scheduler,
        bnm_scheduler = bnm_scheduler,
        viz = viz,
    )

    if args.eval_net:
        start = time.time()
        val_loss, res = trainer.eval_epoch(
            test_loader, is_test=True, obj_id=test_ds.cls_id
        )
        if viz is not None:
            viz.update("val", it, res)
        end = time.time()
        viz.flush()
        print("\nUse time: ", end - start, 's')
    else:
        trainer.train(
            it, start_epoch, config.n_total_epoch, train_loader, val_loader,
            best_loss=best_loss
        )

        if start_epoch == config.n_total_epoch:
            _ = trainer.eval_epoch(val_loader)
