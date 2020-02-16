from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import numpy as np
import tqdm
import sacred
import os.path as osp

from .pytorch_utils import checkpoint_state


if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class _DefaultExCallback(object):
    def __init__(self):
        self.train_vals = {}
        self.train_emas = {}
        self.ema_beta = 0.25

    def __call__(self, ex, mode, k, v):
        # type: (_DefaultExCallback, sacred.Experiment, Any, Any, Any) -> None
        if mode == "train":
            self.train_emas[k] = self.ema_beta * v + (
                1.0 - self.ema_beta
            ) * self.train_emas.get(k, v)
            self.train_vals[k] = self.train_vals.get(k, []) + [v]
            ex.log_scalar("training.{k}".format({"k": k}), self.train_emas[k])

        elif mode == "val":
            ex.log_scalar("val.{k}".format({"k": k}), np.mean(np.array(v)))
            ex.log_scalar(
                "train.{k}".format({"k": k}), np.mean(np.array(self.train_vals[k]))
            )
            self.train_vals[k] = []


class SacredTrainer(object):
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
    eval_frequency : int
        How often to run an eval
    log_name : str
        Name of file to output tensorboard_logger to
    """

    def __init__(
        self,
        model,
        model_fn,
        optimizer,
        lr_scheduler=None,
        bnm_scheduler=None,
        eval_frequency=-1,
        ex=None,
        checkpoint_dir=None,
    ):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model,
            model_fn,
            optimizer,
            lr_scheduler,
            bnm_scheduler,
        )
        self.checkpoint_dir = checkpoint_dir

        self.eval_frequency = eval_frequency

        self.ex = ex
        self.update_callbacks = {}
        self.default_cb = _DefaultExCallback()

    def add_callback(self, name, cb):
        self.update_callbacks[name] = cb

    def add_callbacks(self, cbs={}, **kwargs):
        cbs = dict(cbs)
        cbs.update(**kwargs)
        for name, cb in cbs.items():
            self.add_callback(name, cb)

    def _update(self, mode, val_dict):
        for k, v in val_dict.items():
            if k in self.update_callbacks:
                self.update_callbacks[k](self.ex, mode, k, v)
            else:
                self.default_cb(self.ex, mode, k, v)

    def _train_it(self, it, batch):
        self.model.train()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(it)

        if self.bnm_scheduler is not None:
            self.bnm_scheduler.step(it)

        self.optimizer.zero_grad()
        _, loss, eval_res = self.model_fn(self.model, batch)

        loss.backward()
        self.optimizer.step()

        return eval_res

    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = 0.0
        count = 1.0
        for i, data in tqdm.tqdm(
            enumerate(d_loader, 0), total=len(d_loader), leave=False, desc="val"
        ):
            self.optimizer.zero_grad()

            _, loss, eval_res = self.model_fn(self.model, data, eval=True)

            total_loss += loss.item()
            count += 1
            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]

        return total_loss / count, eval_dict

    def train(
        self,
        start_it,
        start_epoch,
        n_epochs,
        train_loader,
        test_loader=None,
        best_loss=1e10,
    ):
        # type: (SacredTrainer, Any, int, int, torch.utils.data.DataLoader, torch.utils.data.DataLoader, float) -> float
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

        eval_frequency = (
            self.eval_frequency if self.eval_frequency > 0 else len(train_loader)
        )

        it = start_it
        with tqdm.trange(
            start_epoch, n_epochs, desc="epochs", dynamic_ncols=True
        ) as tbar, tqdm.tqdm(
            total=eval_frequency, leave=False, desc="train", dynamic_ncols=True
        ) as pbar:

            for epoch in tbar:
                for batch in train_loader:
                    res = self._train_it(it, batch)
                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()

                    if self.ex is not None:
                        self._update("train", res)

                    if (it % eval_frequency) == 0:
                        pbar.close()

                        if test_loader is not None:
                            val_loss, res = self.eval_epoch(test_loader)

                            if self.ex is not None:
                                self._update("val", res)

                            if self.checkpoint_dir is not None:
                                is_best = val_loss < best_loss
                                best_loss = min(val_loss, best_loss)

                                state = checkpoint_state(
                                    self.model, self.optimizer, val_loss, epoch, it
                                )

                                name = osp.join(self.checkpoint_dir, "checkpoint.pt")
                                torch.save(state, name)
                                if self.ex is not None:
                                    self.ex.add_artifact(name)

                                if is_best:
                                    name = osp.join(self.checkpoint_dir, "best.pt")
                                    torch.save(state, name)
                                    if self.ex is not None:
                                        self.ex.add_artifact(name)

                        pbar = tqdm.tqdm(
                            total=eval_frequency,
                            leave=False,
                            desc="train",
                            dynamic_ncols=True,
                        )
                        pbar.set_postfix(dict(total_it=it))

        return best_loss
