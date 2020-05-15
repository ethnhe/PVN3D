#!/bin/bash
tst_mdl=train_log/ycb/checkpoints/pvn3d_best.pth.tar
python3 -m demo -checkpoint $tst_mdl -dataset ycb
