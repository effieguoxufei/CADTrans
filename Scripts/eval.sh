#!/bin/bash\

# sample surface point cloud (please convert obj format to stl & step first)
python Model/train/sample_points.py --in_dir result/random_eval --out_dir pcd 

# run evaluation script
CUDA_VISIBLE_DEVICES=0 python Model/train/eval_cad.py --fake result/random_eval --real data/testset
CUDA_VISIBLE_DEVICES=0 python Model/train/eval_seq.py --novel  result/novel.txt --unique   result/unique.txt --data_folde result/visual --output_folder  result/visual_mask/ --coutpair_file result/pair.txt



