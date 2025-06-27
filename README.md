# CADTrans: A Code Tree-Guided CAD Generative Transformer Model with Regularized Discrete Codebooks

[![webpage](https://img.shields.io/badge/🌐-Website%20-blue.svg)](https://effieguoxufei.github.io/CADtrans/) 

Xufei Guo, Xiao Dong,
Juan Cao, Zhonggui Chen

![cadtrans](resources/figure0.png)

> We propose a novel CAD model generation network called CADTrans which is based on a code tree-guided transformer framework to autoregressively generate CAD construction sequences.
> - Firstly, three regularized discrete codebooks are extracted through vector quantized adversarial learning, with each codebook respectively representing  the features of Loop, Profile, and Solid.
> - Secondly, these codebooks  are used to normalize a CAD construction sequence into a structured code tree representation  which is then used to  train a standard transformer network to reconstruct the code tree.
> - Finally, the code tree is used as global information to guide the sketch-and-extrude method to recover the corresponding geometric information, thereby reconstructing the complete CAD model.


## Project Requirements 📋

### Environment & Dependencies 🛠️
- Linux 🐧
- Python 3.8 🐍
- PyTorch ≥ 1.10 🔥
- CUDA ≥ 11.4 ⚡

Create and activate the environment:
```
conda create --name cadtrans_env python=3.8 -y
conda activate cadtrans_env
```

- Install pythonocc following the instruction [here](https://github.com/tpaviot/pythonocc-core) (use mamba if conda is too slow).
- Install PyTorch and other dependencies.
```
pip install -r requirements.txt
```

## Data 🗂️
Download our [raw data](https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/cad_raw.rar?download=true), processed from [DeepCAD](https://github.com/ChrisWu1997/DeepCAD), into the `data` folder in the root of this repository.

The raw data need to be first converted to CADTrans format following the steps from [SkexGen](https://github.com/samxuxiang/SkexGen). You can also run the following script to process the data:

```
sh scripts/data_process.sh
```

Alternatively, you can download the already [pre-processed data](https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/model.rar?download=true,https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/loop.rar?download=true,https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/solid.rar?download=true,https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/profile.rar?download=true)🤗

## Training 🏃‍♂️
### Regularized Discrete Codebooks 📚
Train and extract the regularized solid codebooks with:

```
sh scripts/solid_codebook.sh
```

Train and extract the regularized profile codebooks with:

```
sh scripts/profile_codebook.sh
```

Train and extract the regularized loop codebooks with:

```
sh scripts/loop_codebook.sh
```

Download our pretrained checkpoint and extract codes as follows:

| Name     | Checkpoint | Codebook |
|----------|------------|----------|
| Solid    | [solid.pt](https://huggingface.co/datasets/rainforeast/CADTrans/tree/main/proj_log/solid)        | [solid.pkl](https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/solid.pkl?download=true)       |
| Profile  | [profile.pt](https://huggingface.co/datasets/rainforeast/CADTrans/tree/main/proj_log/profile)         | [profile.pkl](https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/profile.pkl?download=true)       |
| Loop     | [loop.pt](https://huggingface.co/datasets/rainforeast/CADTrans/tree/main/proj_log/loop)         | [loop.pkl](https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/loop.pkl?download=true)       |

### Code Tree Generation 🌳
Train code tree with:

```
sh scripts/train_codetree.sh
```

Download our pretrained checkpoint [code_tree.pt](https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/proj_log/code_tree.rar?download=true).🤗

### CAD Generation 🛠️
Train CAD Construction Sequence Generation with:

```
sh scripts/train_cad.sh
```

Download our pretrained checkpoint [cad_gen.pt](https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/proj_log/cad_gen.rar?download=true).🤗

## Generation and Evaluation 🎨
### Generation
Run this script to generate CAD samples and visualize the results:

```
sh scripts/sample.sh
```

### Evaluation
Please download the [test data](https://huggingface.co/datasets/rainforeast/CADTrans/resolve/main/testset.rar?download=true) inside the `data` folder. Run this script to evaluate CAD samples (Warning: This step can be very slow):

```
sh scripts/eval.sh
```


## Acknowledgement
This work was supported by the National Key R&D Program of China (No. 2022YFB3303400), National Natural Science Foundation of China (Nos. 62272402, 62372389), Natural Science Foundation of Fujian Province (Nos. 2024J01513243, 2022J01001), and Fundamental Research Funds for the Central Universities (No. 20720220037).
        


## Citation
If you find our work useful in your research, please cite the following paper
```
@article {guo2025cadtransn,
title = {CADTrans: A Code Tree-Guided CAD Generative Transformer Model with Regularized Discrete Codebooks},
author = {Guo, Xufei and Dong, xiao and Cao, Juan and Chen, Zhonggui},
journal = {},
year = {2025}
}
```
