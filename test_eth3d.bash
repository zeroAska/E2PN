# Train on full, test on cropped full
python run_eth3d.py experiment --experiment-id rot_s2 --max-rotation-degree 10.0  --run-mode eval --model-dir ckpt/rot_s2/eval/  -d /home/`whoami`/data/eth3d/ -r ckpt/rot_s2/train/rot_s2/model_20231115_22\:05\:16/ckpt/rot_s2_net_best.pth   model --flag permutation --kanchor 12
