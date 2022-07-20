# CyCTR-PyTorch
This is a PyTorch implementation of ECCV 2022 paper "[Adaptive Agent Transformer for Few-shot Segmentation]

# Usage

### Environment
```
Python==3.8
torch==1.8.1
torchvision==0.9.0
```

#### Build Dependencies of Deformable DETR for Representation Encoder
```
cd model/ops/
bash make.sh
cd ../../
```

### Data Preparation

+ PASCAL-5^i  

+ COCO-20^i

+ Please refer to [CyCTR_Pytorch](https://github.com/YanFangCS/CyCTR-Pytorch) to prepare the PASCAL and COCO dataset for few-shot segmentation.

### Train
Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.

Then, run this command: 
```
    sh train.sh "dataset" "model_config"
```

### Test
+ Run the following command: 
```
    sh test.sh "dataset" "model_config"
```

# Acknowledgement

This project is built upon [PFENet](https://github.com/dvlab-research/PFENet) and [CyCTR_Pytorch](https://github.com/YanFangCS/CyCTR-Pytorch), thanks for their great works!
