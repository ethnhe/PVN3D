#!/bin/bash
cls_lst=('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')
cls=${cls_lst[0]}
python3 -m train.train_linemod_pvn3d --cls ${cls}
