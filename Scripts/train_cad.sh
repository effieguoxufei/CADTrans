#!/bin/bash\

# train
python ./Model/train_cad.py --output ./proj_log/gen_cad --batchsize 512 \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --mode uncond --device 0,1