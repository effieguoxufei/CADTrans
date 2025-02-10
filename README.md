# CADTrans: A Code Tree-Guided CAD Generative Transformer Model with Regularized Discrete Codebooks

[![webpage](https://img.shields.io/badge/🌐-Website%20-blue.svg)](https://effieguoxufei.github.io/CADtrans/) 

*[Xufei Guo](), [Xiao Dong](),
[Juan Cao](/), [Zhonggui Chen]()*

![cadtrans](resources/figure0.png)

> We propose a novel CAD model generation network called CADTrans which is based on a code tree-guided transformer framework to autoregressively generate CAD construction sequences.
> - Firstly, three regularized discrete codebooks are extracted through vector quantized adversarial learning, with each codebook respectively representing  the features of Loop, Profile, and Solid.
> - Secondly, these codebooks  are used to normalize a CAD construction sequence into a structured code tree representation  which is then used to  train a standard transformer network to reconstruct the code tree.
> - Finally, the code tree is used as global information to guide the sketch-and-extrude method to recover the corresponding geometric information, thereby reconstructing the complete CAD model.

## Requirements

### Environment & Dependencies
- Linux
- Python 3.8
- PyTorch >= 1.10
- CUDA >= 11.4
- Install pythonocc following the instruction [here](https://github.com/tpaviot/pythonocc-core) (use mamba if conda is too slow).


### Dependencies

Install PyTorch and other dependencies:
```
conda create --name cadtrans_env  python=3.8 -y
conda activate cadtrans_env

pip install -r requirements.txt
```

## Data
TODO

## Training 
TODO

## Generation and Evaluation
TODO

## Pretrained Checkpoint

## Acknowledgement
This work was supported by the National Key R&D Program of China (No. 2022YFB3303400), National Natural Science Foundation of China (Nos. 62272402, 62372389), Natural Science Foundation of Fujian Province (Nos. 2024J01513243, 2022J01001), and Fundamental Research Funds for the Central Universities (No. 20720220037).
        


## Citation
If you find our work useful in your research, please cite the following paper
```
@article{guo2025cadtransn,
title= {CADTrans: A Code Tree-Guided CAD Generative Transformer Model with Regularized Discrete Codebooks},
author= {Guo, Xufei and Dong, xiao and Cao, Juan and Chen, Zhonggui},
journal = {Graphical Models},
year={2025}
}
```
