# Train on full, test on cropped full
#python run_modelnet_rotation.py experiment --experiment-id rot_s2 --run-mode eval --model-dir ckpt/rot_s2/eval/ --crop-ratio 0.05 --max-rotation-degree 60 -d /home/`whoami`/data/modelnet/EvenAlignedModelNet40PC/ -r ckpt/rot_s2/model_20230929_19\:46\:55/ckpt/rot_s2_net_best.pth  model --flag permutation --kanchor 12


# Train on half, test on cropped full
#python run_modelnet_rotation.py experiment --experiment-id rot_s2 --run-mode eval --model-dir ckpt/rot_s2/eval/ --crop-ratio 0.2 -d /home/`whoami`/data/modelnet/EvenAlignedModelNet40PC/ -r ckpt/rot_s2/rot_s2/model_20231005_01:28:59_half_cat/ckpt/rot_s2_net_best.pth  model --flag permutation --kanchor 12


# Train on airplane, test on cropped full
python run_modelnet_rotation.py experiment --experiment-id rot_s2 --modelnet-airplane-only --run-mode eval --model-dir ckpt/rot_s2/eval/ --crop-ratio 0.2 --max-rotation-degree 70 -d /home/`whoami`/data/modelnet/EvenAlignedModelNet40PC/ -r ckpt/rot_s2/model_20230925_03\:51\:52/ckpt/rot_s2_net_best.pth  model --flag permutation --kanchor 12


#python run_modelnet_rotation.py experiment --experiment-id rot_s2 --crop-ratio 0.2 --run-mode eval --model-dir ckpt/rot_s2/eval/  -d /home/`whoami`/data/modelnet/EvenAlignedModelNet40PC/ -r ckpt/rot_s2/rot_s2/model_20231005_15\:23\:02/ckpt/rot_s2_net_best.pth  model --flag permutation --kanchor 12
