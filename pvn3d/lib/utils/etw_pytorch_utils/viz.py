from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import numpy as np
# import visdom
import time
from tqdm import tqdm
import collections

__all__ = ["VisdomViz", "CmdLineViz"]


class _DefaultVizCallback(object):
    def __init__(self):
        self.train_vals = {}
        self.train_emas = {}
        self.ema_beta = 0.25

    def __call__(self, viz, mode, it, k, v):
        if mode == "train":
            self.train_emas[k] = self.ema_beta * v + (
                1.0 - self.ema_beta
            ) * self.train_emas.get(k, v)
            self.train_vals[k] = self.train_vals.get(k, []) + [v]
            viz.append_element("train", it, self.train_emas[k], k)

        elif mode == "val":
            viz.append_element(k, it, np.mean(np.array(v)), "val")
            if k in self.train_vals.keys():
                viz.append_element(k, it, np.mean(np.array(self.train_vals[k])), "train")
                self.train_vals[k] = []


class VisdomViz(object):
    def __init__(self, env_name="main", server="http://localhost", port=8097):
        print("=====>")
        print("Initializing visdom env [{}]".format(env_name))
        print("server: {}, port: {}".format(server, port))

        self.default_vzcb = _DefaultVizCallback()

        self.viz = visdom.Visdom(
            server=server, port=port, env=env_name, use_incoming_socket=False
        )
        self.wins = {}
        self.update_callbacks = {}
        self.last_update_time = 0
        self.update_interval = 1.0
        self.update_cache = {}
        print("<=====")

    def text(self, _text, win=None):
        self.viz.text(_text, win=win)

    def update(self, mode, it, eval_dict):
        for k, v in eval_dict.items():
            if k in self.update_callbacks:
                self.update_callbacks[k](self, mode, it, k, v)
            else:
                self.default_vzcb(self, mode, it, k, v)

    def add_callback(self, name, cb):
        self.update_callbacks[name] = cb

    def add_callbacks(self, cbs={}, **kwargs):
        cbs = dict(cbs)
        cbs.update(**kwargs)
        for name, cb in cbs.items():
            self.add_callback(name, cb)

    def append_element(self, window_name, x, y, line_name, xlabel="iterations"):
        key = "{}/{}".format(window_name, line_name)
        if key not in self.update_cache:
            self.update_cache[key] = ([x], [y], xlabel)
        else:
            x_prev, y_prev, _ = self.update_cache[key]
            self.update_cache[key] = (x_prev + [x], y_prev + [y], xlabel)

        if time.perf_counter() - self.last_update_time > self.update_interval:
            for k, v in self.update_cache.items():
                win_name, line_name = k.split("/")
                x, y, xlabel = v
                self._append_element(win_name, x, y, line_name, xlabel)

            self.last_update_time = time.perf_counter()
            self.update_cache = {}

    def _append_element(self, window_name, x, y, line_name, xlabel="iterations"):
        r"""
            Appends an element to a line

        Paramters
        ---------
        key: str
            Name of window
        x: float
            x-value
        y: float
            y-value
        line_name: str
            Name of line
        xlabel: str
        """
        if window_name in self.wins:
            self.viz.line(
                X=np.array(x),
                Y=np.array(y),
                win=self.wins[window_name],
                name=line_name,
                update="append",
            )
        else:
            self.wins[window_name] = self.viz.line(
                X=np.array(x),
                Y=np.array(y),
                opts=dict(
                    xlabel=xlabel,
                    ylabel=window_name,
                    title=window_name,
                    marginleft=30,
                    marginright=30,
                    marginbottom=30,
                    margintop=30,
                    legend=[line_name],
                ),
            )

    def flush(self):
        pass


class _DefaultCmdLineCallback(object):
    def __init__(self):
        self.train_vals = {}

    def __call__(self, viz, mode, it, k, v):
        if mode == "train":
            self.train_vals[k] = self.train_vals.get(k, []) + [v]

        elif mode == "val":
            if k in self.train_vals.keys():
                viz.append_element(k, it, np.mean(np.array(self.train_vals[k])), "train")
                self.train_vals[k] = []
            viz.append_element(k, it, np.mean(np.array(v)), "val")


class CmdLineViz(object):
    def __init__(self):
        self.default_vzcb = _DefaultCmdLineCallback()
        self.update_callbacks = {}
        self.flush_vals = collections.OrderedDict()

    def text(self, _text):
        print(_text)

    def update(self, mode, it, eval_dict):
        for k, v in eval_dict.items():
            if k in self.update_callbacks:
                self.update_callbacks[k](self, mode, it, k, v)
            else:
                self.default_vzcb(self, mode, it, k, v)

    def add_callback(self, name, cb):
        self.update_callbacks[name] = cb

    def add_callbacks(self, cbs={}, **kwargs):
        cbs = dict(cbs)
        cbs.update(**kwargs)
        for name, cb in cbs.items():
            self.add_callback(name, cb)

    def append_element(self, window_name, x, y, line_name):
        if not window_name in self.flush_vals:
            self.flush_vals[window_name] = collections.OrderedDict()

        self.flush_vals[window_name][line_name] = y

    def flush(self):
        if len(self.flush_vals) == 0:
            return

        longest_win_name = max(map(lambda k: len(k), self.flush_vals.keys()))

        tqdm.write("=== Training Progress ===")

        for win, lines in self.flush_vals.items():
            if len(lines) == 0:
                continue

            _str = "{:<{width}} --- ".format(win, width=longest_win_name)
            for k, v in lines.items():
                _str += "{}: {:.4f}\t".format(k, v)

            tqdm.write(_str)

        tqdm.write(" ")
        tqdm.write(" ")

        self.flush_vals = collections.OrderedDict()
