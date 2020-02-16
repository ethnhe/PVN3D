from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import numpy as np
import visdom
import time
from sacred.observers import RunObserver
import pprint


class VisdomObserver(RunObserver):
    def __init__(self, env_name="main", *, server="http://localhost", port=8097):
        super(VisdomObserver, self).__init__()
        self.env_name, self.server, self.port = (env_name, server, port)

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        if "env_name" in config:
            self.env_name = config["env_name"]
        print("=====>")
        print("Initializing visdom env [{}]".format(self.env_name))
        print("server: {}, port: {}".format(self.server, self.port))

        self.viz = visdom.Visdom(
            server=self.server,
            port=self.port,
            env=self.env_name,
            use_incoming_socket=False,
        )
        self.wins = {}
        print("<=====")

        self.viz.text(
            pprint.pformat(
                (dict(host_info=host_info, start_time=start_time, config=config))
            ),
            win=None,
        )

    def log_metrics(self, metrics_by_name, info):
        for key in metrics_by_name:
            mode, metric_name = key.split(".")
            if mode == "training":
                self._append_element(
                    "training",
                    metrics_by_name[key]["steps"],
                    metrics_by_name[key]["values"],
                    metric_name,
                    "iter",
                )
            elif mode == "val" or mode == "train":
                self._append_element(
                    metric_name,
                    metrics_by_name[key]["steps"],
                    metrics_by_name[key]["values"],
                    mode,
                    "iter",
                )

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
                opts=dict(showlegend=True),
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
                    showlegend=True,
                ),
            )
