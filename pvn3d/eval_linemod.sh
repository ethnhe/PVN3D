sv_img=False

echo iron > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/iron/kpnet_model_195_0.40526514861873514.pth

echo glue > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*16)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/glue/kpnet_model_236_0.17613368513041022.pth
# 
echo driller > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/driller/kpnet_model_197_0.35828931267835695.pth

echo phone > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/phone/kpnet_model_266_0.2533587273442894.pth

echo ape > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*16)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/ape/kpnet_model_448_0.1282156617300851.pth

echo benchvise > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/benchvise/kpnet_model_235_0.28707761588946795.pth
# 
echo cam > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/cam/kpnet_model_351_0.2781084878771913.pth
# # 
echo cat > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/cat/kpnet_model_356_0.15835757288866176.pth

echo duck > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/duck/kpnet_model_403_0.14983670655550532.pth
# 
echo eggbox > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/eggbox/kpnet_model_347_0.27863180715713143.pth
# 
echo lamp > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/lamp/kpnet_model_259_0.2916407694972179.pth
# 
echo holepuncher > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*6)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/holepuncher/kpnet_model_355_0.22879577885345545.pth

echo can > cls_type.txt
rlaunch --cpu=10 --gpu=1 --memory=$((1024*16)) -- python3 eval.py --stragety 1 --save_ins $sv_img --test_net ./train_log/models/can/kpnet_model_260_0.24537508952336048.pth

