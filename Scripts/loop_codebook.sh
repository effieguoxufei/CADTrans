#!/bin/bash\

# LOOP CODEBOOK
python ./Model/codebook/codebook.py --output proj_log/loop --batchsize 256 --format loop --device 0
python ./Model/codebook/extract_code.py --checkpoint proj_log/loop --format loop --epoch 250 --device 0


