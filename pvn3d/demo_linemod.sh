#!/bin/bash

# cls_lst=('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')

cls='ape'
tst_mdl=train_log/linemod/checkpoints/${cls}/${cls}_pvn3d_best.pth.tar
python3 -m demo -dataset linemod -checkpoint $tst_mdl -cls $cls
