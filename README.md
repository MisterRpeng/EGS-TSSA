# EGS-TSSA

Code for the method [Transferable Structural Sparse Adversarial Attack Via Exact Group Sparsity Training]

## Dependencies
1. Install [pytorch](https://pytorch.org/). This repo is tested with pytorch=2.0.0, python=3.10.10.
2. Install python packages using following command:
```
pip install -r requirements.txt
```


## Pretrained-Generators
Download pretrained adversarial generators are available in the `/EGS_TSAA/weights/` directory.

Adversarial generators are trained against following two models.
* Inception-V3
* ResNet50
```
weights/hard_eps10_incv3_tk0.95.pth
weights/hard_eps10_res50_tk0.82.pth
weights/soft_eps255_incv3_tk1.pth
weights/soft_eps255_res50_tk0.873.pth
weights/target971_eps255_incv3_tk1.pth
weights/target971_eps255_res50_tk0.7.pth
```

These models are trained on ImageNet and available in Pytorch. 
  
## Datasets
* Training data:
  * [ImageNet](http://www.image-net.org/) Training Set.
  
* Evaluations data:
  * randomly selected 5k images from [ImageNet](http://www.image-net.org/) Validation Set.
  * The test data is the same as that of the TSAA method, you can download evaluations data from [here](https://drive.google.com/drive/folders/1z6fMGd-NFvKi1-tVG59ow7ZxHyEGfEGI?usp=sharing), shared by TSAA authors.
  
  
## Training
<p align="justify"> Run the following command

```
  python train.py --train_dir [path_to_train] --model_type res50 --eps 10 --target -1 --tk 0.6
```
This will start trainig a generator trained on one dataset (--train_dir) against ResNet50 (--model_type) under perturbation budget $\ell_\infty$=255 (--eps) and a top-k setting $tk=0.6$ in a non-targeted setting (--target).<p>

## Evaluations
<p align="justify"> Run the following command

```
  python test.py --test_dir [path_to_val] --model_type res50 --model_t vgg16 --eps 10 --target -1 --checkpoint [path_to_checkpoint] --tk 0.6
```
This will load a generator trained against ResNet50 (--model_type) and evaluate clean and adversarial accuracy of VGG16 (--model_t) under perturbation budget 10 (--eps) and a top-k setting $tk=0.6$ in a targeted setting (--target). <p>
The top-k values of the test are affected by the sparsity level. Please modify the top-k according to the value of tk in the training pre-weighting remarks.<p>
For example, if the weights are `hard_eps10_res50_k0.82.pth`, you should set `tk=0.82`.