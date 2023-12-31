# PCDNF: Revisiting Learning-based Point Cloud Denoising via Joint Normal Filtering

:zap:`Status Update: [2023/07/02] This paper has been accepted by the IEEE Transactions on Visualization and Computer Graphics (TVCG).`

<p align='center'>
<img src='image/figure2.png'/>
</p>

 by [Zheng Liu](https://labzhengliu.github.io/), Yaowu Zhao, Sijing Zhan, [Yuanyuan Liu](https://cvlab-liuyuanyuan.github.io/), [Renjie Chen](http://staff.ustc.edu.cn/~renjiec/) and [Ying He](https://personal.ntu.edu.sg/yhe/)

 ## :bulb: Introduction
Recovering high quality surfaces from noisy point clouds, known as point cloud denoising, is a fundamental yet challenging
problem in geometry processing. Most of the existing methods either directly denoise the noisy input or filter raw normals followed by
updating point positions. Motivated by the essential interplay between point cloud denoising and normal filtering, we revisit point cloud
denoising from a multitask perspective, and propose an end-to-end network, named PCDNF, to denoise point clouds via joint normal
filtering. In particular, we introduce an auxiliary normal filtering task to help the overall network remove noise more effectively while
preserving geometric features more accurately. In addition to the overall architecture, our network has two novel modules. On one
hand, to improve noise removal performance, we design a shape-aware selector to construct the latent tangent space representation of
the specific point by comprehensively considering the learned point and normal features and geometry priors. On the other hand, point
features are more suitable for describing geometric details, and normal features are more conducive for representing geometric
structures (e.g., sharp edges and corners). Combining point and normal features allows us to overcome their weaknesses. Thus, we
design a feature refinement module to fuse point and normal features for better recovering geometric information.

<p align='center'>
<img src='image/figure1.png'/>
</p>

## :wrench: Usage
## Environment
* Python 3.6
* PyTorch 1.5.0
* CUDA and CuDNN (CUDA 10.1 & CuDNN 7.5)
* TensorboardX (2.0) if logging training info.
## Install required python packages:
``` bash
pip install numpy
pip install scipy
pip install plyfile
pip install scikit-learn
pip install tensorboardX (only for training stage)
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
### Test the trained model:
Set the parameters such as file path, batchsize, iteration numbers, etc in **testN.py** and then run it.
We provide our pretrained model.

### Train the model:
Set the parameters such as file path, batchsize, iteration numbers, etc in **train_NetworkN1.py** and then run it.
Our training set is from [PointFilter](https://github.com/dongbo-BUAA-VR/Pointfilter) and the normal information is computed by PCA.

## :link: Citation
If you find this work helpful please consider citing our [paper](https://ieeexplore.ieee.org/document/10173632) :
```
@ARTICLE{10173632,
  author={Liu, Zheng and Zhao, Yaowu and Zhan, Sijing and Liu, Yuanyuan and Chen, Renjie and He, Ying},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={PCDNF: Revisiting Learning-based Point Cloud Denoising via Joint Normal Filtering}, 
  year={2023},
  doi={10.1109/TVCG.2023.3292464}
}
```

