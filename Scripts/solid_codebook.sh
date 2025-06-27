#!/bin/bash\

# SOLID CODEBOOK
python ./Model/codebook/codebook.py --output proj_log/solid --batchsize 256 --format solid --device 0
python ./Model/codebook/extract_code.py --checkpoint proj_log/solid --format solid --epoch 250 --device 0


