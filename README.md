# EGS-TSSA
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ming_Transferable_Structural_Sparse_Adversarial_Attack_Via_Exact_Group_Sparsity_Training_CVPR_2024_paper.pdf)] [[Supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Ming_Transferable_Structural_Sparse_CVPR_2024_supplemental.pdf)] [[Poster](https://drive.google.com/file/d/1QqOaq-PXrNTeZvV9oVvoANz2NsqMTOst/view?usp=sharing)] [[Presentation](https://drive.google.com/file/d/1tE_1PUYNf5KXrjoD3VPbCmNM7cJI-2Uq/view?usp=drive_link)]

The official implementation of [**\[CVPR 2024\] "Transferable Structural Sparse Adversarial Attack Via Exact Group Sparsity Training", Di Ming, Peng Ren, Yunlong Wang, Xin Feng\***](https://openaccess.thecvf.com/content/CVPR2024/html/Ming_Transferable_Structural_Sparse_Adversarial_Attack_Via_Exact_Group_Sparsity_Training_CVPR_2024_paper.html). 

## Introduction
Deep neural networks (DNNs) are vulnerable to highly transferable adversarial attacks. Especially, many studies have shown that sparse attacks pose a significant threat to DNNs on account of their exceptional imperceptibility. Current sparse attack methods mostly limit only the magnitude and number of perturbations while generally overlooking the location of the perturbations, resulting in decreased performances on attack transferability. A subset of studies indicates that perturbations existing in the significant regions with rich classification-relevant features are more effective. Leveraging this insight, we introduce the structural sparsity constraint in the framework of generative models to limit the perturbation positions. To ensure that the perturbations are generated towards classification-relevant regions, we propose an exact group sparsity training method to learn pixel-level and group-level sparsity. For purpose of improving the effectiveness of sparse training, we further put forward masked quantization network and multi-stage optimization algorithm in the training process. Utilizing CNNs as surrogate models, extensive experiments demonstrate that our method has higher transferability in image classification attack compared to state-of-the-art methods at approximately same sparsity levels. In cross-model ViT, object detection, and semantic segmentation attack tasks, we also achieve a better attack success rate. 

![Home](https://github.com/MisterRpeng/EGS-TSSA/blob/main/show_image/Home.png)


# Getting Started

## Dependencies
1. Install [pytorch](https://pytorch.org/). This repo is tested with pytorch=2.0.0, python=3.10.10.
2. Install python packages using following command:
```
pip install -r requirements.txt
```


## Pretrained-Generators
Pretrained adversarial generators can be downloaded from the provided [link](https://drive.google.com/drive/folders/1iypmK4iIdR2drG6EEe0f2K66fJRC9i7N?usp=sharing), and can be placed under the `/EGS_TSSA/weights/` directory folder to generate structural sparse adversarial perturbation on test sample.

Adversarial generators are trained against the following two models.
* Inception-V3
* ResNet50
```
weights/soft_eps10_incv3_tk1.pth
weights/soft_eps10_res50_tk0.95.pth
weights/hard_eps10_incv3_tk0.95.pth
weights/hard_eps10_res50_tk0.82.pth
weights/soft_eps255_incv3_tk1.pth
weights/soft_eps255_res50_tk0.873.pth
weights/hard_eps255_incv3_tk0.9.pth
weights/hard_eps255_res50_tk0.671.pth
weights/target971_eps255_incv3_tk1.pth
weights/target971_eps255_res50_tk0.7.pth
```

These models are trained on ImageNet and available in Pytorch. 
  
### Datasets
* Training data:
  * [ImageNet](http://www.image-net.org/) Training Set.
  
* Evaluations data:
  * Randomly select 5k images from the [ImageNet](http://www.image-net.org/) Validation Set.
  * The test data is the same as that used in the TSAA method, and you can download the evaluation data from [here](https://drive.google.com/drive/folders/1z6fMGd-NFvKi1-tVG59ow7ZxHyEGfEGI?usp=sharing) provided by TSAA authors.
  
  
### Training
<p align="justify"> Run the following command

```
  python train.py --train_dir [path_to_train] --model_type res50 --eps 10 --target -1 --tk 0.6
```
This will start training a generator on the given dataset (--train_dir) against ResNet50 (--model_type), under the constraint of the perturbation budget $\ell_\infty$=10 (--eps) and the top-k selection $tk=0.6$, in the non-targeted attack setting (--target).<p>

### Evaluations
<p align="justify"> Run the following command

```
  python test.py --test_dir [path_to_val] --model_type res50 --model_t vgg16 --eps 10 --target -1 --checkpoint [path_to_checkpoint] --tk 0.6
```
This will load a generator trained against ResNet50 (--model_type), and evaluate the attack success rate (ASR) on the target model VGG16 (--model_t), under the constraint of the perturbation budget 10 (--eps) and the top-k selection $tk=0.6$, in the non-targeted attack setting (--target). <p>
The top-k values of the test are affected by the sparsity level. Please modify the top-k (--tk) according to the value of tk mentioned in the pretrained weight file.<p>

For example, if the weight file is `hard_eps10_res50_tk0.82.pth`, you should set `tk=0.82`.

## Acknowledgements
The code refers to  [TSAA](https://github.com/shaguopohuaizhe/TSAA), [StrAttack](https://github.com/KaidiXu/StrAttack).

We thank the authors for sharing sincerely.

## Citation
If you find this work is useful in your research, please cite our paper:
```
@InProceedings{CVPR24_EGS_TSSA,
    author    = {Ming, Di and Peng, Ren and Wang, Yunlong and Feng, Xin},
    title     = {Transferable Structural Sparse Adversarial Attack Via Exact Group Sparsity Training},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {24696-24705}
}
```
## Contact

[Peng Ren](https://github.com/MisterRpeng/): [MisterR_2019@163.com](mailto:MisterR_2019@163.com)

[Di Ming](https://midasdming.github.io/): [diming@cqut.edu.cn](mailto:diming@cqut.edu.cn)
