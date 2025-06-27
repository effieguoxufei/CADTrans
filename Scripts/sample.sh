#!/bin/bash\

# sample code-tree & CAD model
python Model/train/random_gen.py --code_weight proj_log/code_tree  --cad_weight proj_log/cad_gen --output result/random_eval \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --novel_pair_file result/novel.txt \
    --unique_pair_file result/unique.txt --coutpair_file result/novel.txt --mode eval --device 1
python Model/train/convert.py --data_folder result/random_eval


# visualize CAD 
python Model/train/cad_img.py --input_dir result/random_eval --output_dir  result/visual
