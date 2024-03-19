# EGS-TSSA
[[Paper]()] [[Supp]()] [[Poster]()] [[Presentation]()]

The official implementation of [**\[CVPR 2024\] "Transferable Structural Sparse Adversarial Attack Via Exact Group Sparsity Training", Di Ming, Peng Ren, Yunlong Wang, Xin Feng\***](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers). 

## Introduction
Highly transferable adversarial attacks pose a significant threat to deep neural networks (DNNs). Many studies have shown that DNNs are also vulnerable to sparsity attacks. Current sparse attack methods mostly limit only the magnitude and number of perturbations, while generally overlooking the location of the perturbations. A subset of studies, however, indicates that perturbations existing in the significant regions with rich classification-relevant features are more effective. This suggests that incorporating appropriate positional information can lead to the generation of perturbations in a more efficient direction. Leveraging this insight, we introduce structural sparsity constraints in the perturbation generation process to limit the perturbation positions. To ensure that the perturbations are generated towards the classification of important regions, we propose an exact group sparsity training method to implement structural sparsity constraints according to group feature importance. To improve the effectiveness of sparse training, we introduce a masked quantization network and multi-stage optimization strategy in the training process. With CNNs employed as surrogate models, extensive experiments showed that our method had higher transferability and perturbation imperceptibility in the approximate sparsity setting as compared to state-of-the-art methods. In cross-model ViT, object detection, and semantic segmentation attack tasks, we also achieved a higher attack success rate.

<table>
  <tr>
    <td> <div align="center"><img src=https://github.com/MisterRpeng/EGS-TSSA/blob/main/show_image/Flowchart.jpg width=30% /></div> </td>
    <td> <div align="center"><img src=https://github.com/MisterRpeng/EGS-TSSA/blob/main/show_image/Comparison_of_adversarial_perturbations.jpg width=100% /></div> </td>
  </tr>
  <tr>
    <td> The overall framework of our proposed transferable structural sparse attack method (Top: generative network $G$; Bottom: masked quantization network $Q$)  </td>
    <td> Comparison of perturbation pattern across different adversarial attack methods. Our EGS-TSSA method produces perturbations that are noticeably more concentrated and structured as compared to other methods. </td>
  </tr>
</table>


## Methodology
<div align="center"><img src=https://github.com/MisterRpeng/EGS-TSSA/blob/main/show_image/Flowchart.jpg width=65% /></div>

The overall framework of our proposed transferable structural sparse attack method (Top: generative network $G$; Bottom: masked quantization network $Q$) 

## Demo (Adversarial Examples & Perturbations)
<div align="center"><img src=https://github.com/MisterRpeng/EGS-TSSA/blob/main/show_image/Comparison_of_adversarial_perturbations.jpg width=80% /></div>

Comparison of perturbation pattern across different adversarial attack methods. Our EGS-TSSA method produces perturbations
that are noticeably more concentrated and structured as compared to other methods.


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
    pages     = {}
}
```
## Contact

[Peng Ren](https://github.com/MisterRpeng/): [MisterR_2019@163.com](mailto:MisterR_2019@163.com)

[Di Ming](https://midasdming.github.io/): [diming@cqut.edu.cn](mailto:diming@cqut.edu.cn)
