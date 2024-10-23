## Dataset

The overall directory structure should be:
```
│PCP-MAE/
├──cfgs/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
│   ├──Stanford3dDataset_v1.2_Aligned_Version/
│   ├──s3dis/
├──datasets/
├──.......
```

### ModelNet40 Dataset: 

```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```
Download: You can download the processed data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or download from the [official website](https://modelnet.cs.princeton.edu/#) and process it by yourself. For the three text files (modelnet40_shape_names.txt, modelnet40_train.txt and modelnet40_test.txt), please check the `data` directory in the [Point-BERT repo](https://github.com/Julie-tang00/Point-BERT/tree/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/data).

### ModelNet Few-shot Dataset:
```
│ModelNetFewshot/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.

### ScanObjectNN Dataset:
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

### ShapeNet55/34 Dataset:

```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  ├── .......
├──ShapeNet-55/
│  ├── train.txt
│  └── test.txt
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md).

### ShapeNetPart Dataset:

```
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```

Download: Please download the data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). 

### S3DIS Dataset:

```shell
|Stanford3dDataset_v1.2_Aligned_Version/
├──Area_1/
│  ├── conferenceRoom_1
│  ├── .......
│── .......
│stanford_indoor3d
│──Area_1_conferenceRoom_1.npy
│──Area_1_office_19.npy
```
Please prepare the dataset following [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch):
download the `Stanford3dDataset_v1.2_Aligned_Version` from [here](http://buildingparser.stanford.edu/dataset.html), and get the processed `stanford_indoor3d` with:

```shell
cd ./semantic_segmentation/data_utils
python collect_indoor3d_data.py
```
