# Train on full, test on cropped full
python run_eth3d.py experiment --experiment-id rot_s2 --max-rotation-degree 10.0  --run-mode train --model-dir ckpt/rot_s2/train/  -d /home/`whoami`/data/eth3d/   model --flag permutation --kanchor 12
