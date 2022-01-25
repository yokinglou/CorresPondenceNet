# CorresPondenceNet (CPNet)

This repository includes implementation and dataset for the ECCV '20 paper: "Human Correspondence Consensus for 3D Object Semantic Understanding". You can access the full paper from [here](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670494.pdf) or [arXiv](https://arxiv.org/abs/1912.12577).

![intro](/figs/intro.png)
<!-- <img src="/figs/intro.png" width="600"/> -->

## Dataset
CPNet dataset has a collection of 25 categories, 2,334 models based on ShapeNetCore, which includes 1,000+ correspondence sets with 104,861 points. The correspondence sets ![1](http://latex.codecogs.com/svg.latex?\\{\\mathcal{C}_i|i=1,\\cdots,N_{\\mathcal{C}}\\}) are shwon in figure below. Each correspondence ![2](http://latex.codecogs.com/svg.latex?\\mathcal{C}_i) contains points with same semantics on each object.
![corr_sets](/figs/corr_sets.jpg)
<!-- <img src="/figs/corr_sets.jpg" width="600"/> -->

Dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1C3lOg8rmNOVvMc-1lLVShkbw95zs5sBC/view?usp=sharing).

### Data Layout
CPNet dataset contains the correspondence annotations on each object and geodesic distances between correspondences. The structure of dataset is as follow:
```
<data_path>/
    name2id.json
    <class_id>.h5
    corr_mean_dist_geo/
        <class_id>_mean_distances.h5
```
The details of each file is as follows:

- name2id.json: The dictionary between name and id of each class

- <class_id>.h5
    - "point_clouds": point coordinates of point cloud
    - "keypoints": point indexes of different correspondences
    - "mesh_names": names of meshes in ShapeNet

- <class_id>_mean_distances.h5
    - "mean_distance": distances between correspondences

## How to Learn Dense Semantic Embeddings
![embedding](/figs/embedding.png)

### Install
```
# Create a new conda environment
conda create -n cpnet python=3.7
conda activate cpnet

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

### Data
Please add the absolute path of the CPNet dataset in [config](./config/config.yaml) at first. 

### Traing a Model
You can train on CPNet and get pointwise semantic embeddings with different backbones in [models](./models/), for example:
```
python train.py network=pointnet2 class_name=airplane
```
### Testing with mGE
If you want to test the trained model for correspondence benchmark, please run
```
python test.py network=pointnet2 class_name=airplane
```

## Citing CPNet
If you use CPNet in your research, please cite the [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670494.pdf):
```
@inproceedings{lou2020human,
    title={Human Correspondence Consensus for 3D Object Semantic Understanding},
    author={Lou, Yujing and You, Yang and Li, Chengkun and Cheng, Zhoujun and Li, Liangwei and Ma, Lizhuang and Wang, Weiming and Lu, Cewu},
    booktitle={European Conference on Computer Vision},
    pages={496--512},
    year={2020},
    organization={Springer}
}
```

## References
The implementations of models is referenced by:
- [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn)
- [DylanWusee/pointconv_pytorch](https://github.com/DylanWusee/pointconv_pytorch)
- [sausagecy/RSCNN_Pytorch1.0](https://github.com/sausagecy/RSCNN_Pytorch1.0)
- [qianguih/RSNet](https://github.com/qianguih/RSNet)
- [xyf513/SpiderCNN](https://github.com/xyf513/SpiderCNN)
