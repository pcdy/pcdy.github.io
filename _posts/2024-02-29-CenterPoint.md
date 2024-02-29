---
layout: post
title: Center Point
date: 2024-02-29
categories: object detection
tags: point cloud
---

# Center Point： Center-based 3D Object Detection and Tracking - 论文理解+部分代码

## 1. 基本信息

**Authors: Tianwei Yin et al.**, **德克萨斯大学奥斯汀分校**

**CVPR 2021**

![image-20230529225922991](https://huatu.98youxi.com/markdown/work/uploads/upload_30723195b66d9a5d981cd9344b2b5d38.png)


**支持框架：**

**1. Paddle3D** **https://github.com/PaddlePaddle/Paddle3D**

**2. MMdetection3D** **https://github.com/open-mmlab/mmdetection3d**

**3. OpenPCDet** **https://github.com/open-mmlab/OpenPCDet**

**4. Official Implement** **https://github.com/tianweiy/CenterPoint**

## 2. Motivation

1. Box-based 方法**直接预测**目标尺寸和方向，枚举数量多；

   是否可以**先预测目标及中心**，**再预测目标尺寸和方向**？

![](https://huatu.98youxi.com/markdown/work/uploads/upload_4d9dfc264658805377b4459268b0ec2e.png)

2. CenterNet 《Objects as Points》, 2019 CVPR, 引用数2778.

   作者前序工作在图像目标检测中提出了CenterNet，可扩展到3维Point Cloud上。

## 3. Framework

CenterPoint模型框架如下图所示，包括三部分内容：

![Uploading file..._5oz2qse67](https://huatu.98youxi.com/markdown/work/uploads/upload_cca04fa0031f4cebbdf3b3f81c304d89.png)


1. 3D Backbone，提取点云特征，并映射到二维。文中采用两种方式：VoxelNet和PointPillars方式。此部分内容采用两种典型的3D特征提取主干模型，不是论文创新部分。
2. Head，采用类似centerNet的方式提取目标中心，基于中心再回归目标的属性，如尺寸，朝向等信息。
3. refine stage。对检测的目标框进行优化，跟多数二阶段检测方法类似。

### 3.1 3D Backbone

![](https://huatu.98youxi.com/markdown/work/uploads/upload_01dff2be30e768784e083bd57ee79c3d.png)


文中采用的两种3D主干网络如上图所示，分别用了VoxelNet和PointPillars的对应部分。以VoxelNet为例对应mmdetection3D代码部分如下：

在文件 `.\mmdetection3d\configs\_base_\models\centerpoint_01voxel_second_secfpn_nus.py`中，定义了相关细节，具体实现见每部分代码。

```python
	#体素化的基本要求（体素内点数量、体素大小、voxel数量）
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size, max_voxels=(90000, 120000)),
    #体素内点云处理方式，HardSimpleVFE
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    #3D稀疏卷积模块
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
```

经过SparseEncoder后，点云特征维度变成(N,C*D,H,W)：N-batch size, C-Channels, D-depth, H-height, W-width。映射为BEV

```python
	N, C, D, H, W = spatial_features.shape
    spatial_features = spatial_features.view(N, C * D, H, W) #直接通道维度和Depth维度转为一维
```

### 3.2 2D detection head

**主要包含两部分：**

1. 2D的backbone, 进一步对BEV下特征进行特征提取。
2. **CenterPoint**的检测头**(Main Contribution)**，以目标中心位置为关键点进行关键点检测，生成**heatmap**，并回归出3D Boundingbox的中心高度，长宽高和航向角等信息。

2D backbone采用的是SECOND的RPN结构，在文件 `centerpoint_01voxel_second_secfpn_nus.py`中具体配置如下：

```python
	#主干特征提取模块 - 1次降采样
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    #FPN 特征金字塔融合多尺度特征 - 1次上采样恢复空间分辨率
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
```

CenterPoint的检测头采用的类似CenterNet的检测头，在文件 `centerpoint_01voxel_second_secfpn_nus.py`中具体配置如下：

```python
 pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
     #分成多个类别分别预测
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
     #GT编码方式
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
     #对每个类别的生成的检测头
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
     # 分类和bbox的代价函数
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
```

这部分主要包括两部分，1是真值的处理部分（真值标签为边界框和类别），2是模型的前向预测部分（预测结果为heatmap等），这两部分必须匹配才能进行代价计算。

以下代码为 `centerpoint_head.py`中CenterHead的forward部分，为预测结果：

```python
def forward(self, feats):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats)
  
# 其中forward_single为：
def forward_single(self, x):
        """Forward function for CenterPoint.
        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].
        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []
        x = self.shared_conv(x)
        for task in self.task_heads: #SeperateHead部分的，完成每个头的处理和输出
            ret_dicts.append(task(x))
        return ret_dicts
# 然后SeperateHead部分输出为：
"""Forward function for SepHead.
        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].
        Returns:
            dict[str: torch.Tensor]: contains the following keys:
                -reg （torch.Tensor): 2D regression value with the shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of [B, N, H, W].
"""
```

真值**原始的数据标注为bbox+cls**，因此要进行对应编码，与预测结果对应，生成heatmap，其编码代码为 `centerpoint_head.py`中CenterHead的get_targets部分，其内部又对每个类调用get_targets_single函数：

```python
def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.
                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """
### 具体实现省略
```

**如何生成真值的Heatmap？**

heatmap关键点用二维高斯核表示，根据三种不同情况，基于IoU和真值的长宽设计heatmap关键点半径r。

![](https://huatu.98youxi.com/markdown/work/uploads/upload_bd4dc71e121473b261fd4528349e5fae.png)


半径r推导过程见上述链接，代码在 `mmdet3d\core\utils\gaussian.py`的 `gaussian_radius`函数计算：

```python
def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.
    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float, optional): Gaussian_overlap. Defaults to 0.5.
    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size
    # 情况3
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
	# 情况2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
	# 情况1
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)
```

然后预测跟真值就可以计算Loss了， Loss采用两种代价：分类-GaussianFocalLoss，bbox-L1Loss。

## 4. Experiments

提出模型在Waymo和Nuscenes都取得较好的结果。

其模型已成为一种基础的主干网络，并在Nuscenes取得相对好的结果。下图为描述有使用centerpoint的模型结果。

![](https://huatu.98youxi.com/markdown/work/uploads/upload_a90e064ae45504c5a950d08d19be6785.png)


其中上图排名第一的模型在整个Lidar模型中以NDS为指标排序，排在第三位，截止2023-06-01.

![](https://huatu.98youxi.com/markdown/work/uploads/upload_dd52a0c4877c2a3a8fdac3a4fa56b806.png)
