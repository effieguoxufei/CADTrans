#!/bin/bash\

# PROFILE CODEBOOK
python Model/codebook/codebook.py --output proj_log/profile --batchsize 256 --format profile --device 0
python Model/codebook/extract_code.py --checkpoint proj_log/profile --format profile --epoch 250 --device 0

