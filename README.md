# PCP-MAE

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=pcp-mae-learning-to-predict-centers-for-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/few-shot-3d-point-cloud-classification-on-1)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-1?p=pcp-mae-learning-to-predict-centers-for-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/few-shot-3d-point-cloud-classification-on-2)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-2?p=pcp-mae-learning-to-predict-centers-for-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/few-shot-3d-point-cloud-classification-on-3)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-3?p=pcp-mae-learning-to-predict-centers-for-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/few-shot-3d-point-cloud-classification-on-4)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-4?p=pcp-mae-learning-to-predict-centers-for-point)

> [**PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders**](https://arxiv.org/abs/2408.08753) **NeurIPS 2024 spotlight** <br>
> [Xiangdong Zhang](https://scholar.google.com/citations?user=5S-TKKoAAAAJ&hl=zh-CN&oi=sra)\*, [Shaofeng Zhang](https://scholar.google.com/citations?user=VoVVJIgAAAAJ&hl=zh-CN&oi=sra)\* and [Junchi Yan](https://scholar.google.com/citations?user=ga230VoAAAAJ&hl=zh-CN&oi=sra) <br>

[Arxiv](https://arxiv.org/abs/2408.08753)

## PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders

<div  align="center">    
 <img src="./figs/overview.jpg" width = "1100"  align=center />
</div>

## 📰 News

<!-- - 🍾 Oct, 2024: The corresponding checkpoints are released. -->
<!-- - 📌 Oct, 2024: The training and inference code is released. -->
- 💥 Aug, 2024: PCP-MAE is available in [arxiv](https://arxiv.org/abs/2408.08753).
- 🎉 Sept, 2024: [**PCP-MAE**](https://arxiv.org/abs/2408.08753) is accepted by NeurIPS 2024 as **spotlight**.
- 📌 Oct, 2024: The corresponding checkpoints are released in [Google Drive](https://drive.google.com/drive/folders/18E04xV5r4GtjhLGJIc9Ulo1F5DuOTYU6?usp=drive_link) and the code will coming soon.
- 📌 Oct, 2024: The code has been released.


## ✅ TODO List
- [ ] Complete the introduction for the PCP-MAE project.
- [x] Publish the training and inference code.
- [x] Release the checkpoints for pre-training and finetuning.

## 3. PCP-MAE Models
| Task              | Dataset        | Config                                                               | Acc.       | Checkpoints Download                                                                                     |
|-------------------|----------------|----------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------|
| Pre-training      | ShapeNet       | [base.yaml](cfgs/pretrain/base.yaml)                        | N.A.       | [Pre-train](https://drive.google.com/drive/folders/1smQMWBBEdMOXVAzIBs3xCBrcyQDg8_GS?usp=drive_link)           |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/finetune_scan_objbg.yaml)     | 95.52%     | [OBJ_BG](https://drive.google.com/drive/folders/1He3bUfXJ36nwAcGbQE4I9tOUnxjEmfae?usp=drive_link)          |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/finetune_scan_objonly.yaml) | 94.32%     | [OBJ_ONLY](https://drive.google.com/drive/folders/1xuJlAwSYMwc0bTKvnzaoePggMrLqQw3r?usp=drive_link)        |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/finetune_scan_hardest.yaml) | 90.35%     | [PB_T50_RS](https://drive.google.com/drive/folders/1YWJrThywU6G4yoUn4-GvtnHH_bi_Uprp?usp=drive_link)       |
| Classification    | ModelNet40(1k) w/o voting | [finetune_modelnet.yaml](./cfgs/finetune_modelnet.yaml)         | 94.1%      | [ModelNet40_1K](https://drive.google.com/drive/folders/1JqZGKMjisagw6R1L8BwIbiWQLTJAFVjX?usp=drive_link)     |
| Classification    | ModelNet40(1k) w/ voting | [finetune_modelnet.yaml](./cfgs/finetune_modelnet.yaml)         | 94.4%      | [ModelNet40_1K_voting](https://drive.google.com/drive/folders/1YVlGr52OT3IYOmQ4b1AJc-9Xg6cS_GQh?usp=drive_link)     |
| Part Segmentation | ShapeNetPart   | [segmentation](./segmentation)                                       | 86.9% Cls.mIoU | TBD        |
| Scene Segmentation | S3DIS   | [semantic_segmentataion](./semantic_segmentation)                                       | 61.3% mIoU | TBD        |

| Task              | Dataset    | Config                                   | 5w10s (%)  | 5w20s (%)  | 10w10s (%) | 10w20s (%) | Download                                                                                       |
|-------------------|------------|------------------------------------------|------------|------------|------------|------------|------------------------------------------------------------------------------------------------|
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/fewshot.yaml) | 97.4 ± 2.3 | 99.1 ± 0.8 | 93.5±3.7 | 95.9±2.7 | [FewShot](https://drive.google.com/drive/folders/1EVvSeAS47Wx0pFUO2UbBbTWOB00NVX2M?usp=drive_link) |


The checkpoints and logs have been released on [Google Drive](https://drive.google.com/drive/folders/18E04xV5r4GtjhLGJIc9Ulo1F5DuOTYU6?usp=drive_link). To fully reproduce our reported results, we recommend fine-tuning the pre-trained ckpt-300 with different random seeds (typically 8 different seeds) and recording the best performance which is also adopted by other peer methods (e.g. [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and [ReCon](https://github.com/qizekun/ReCon)). Occasionally, ckpt-275 may outperform ckpt-300, so we encourage you to try to fine-tune with both ckpt-300 and ckpt-275.

## Requirements
PyTorch >= 1.7.0 < 1.11.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
# Quick Start
conda create -n pcpmae python=3.10 -y
conda activate pcpmae

# Install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install required packages
pip install -r requirements.txt
```

```
# Install the extensions
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```


## Datasets

We use ShapeNet, ScanObjectNN, ModelNet40, ShapeNetPart and S3DIS in this work. See [DATASET.md](./DATASET.md) for details.

## Pre-training
To pretrain PCP-MAE on ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain/base.yaml --exp_name <output_file_name>
```
## Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model> --seed $RANDOM
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model> --seed $RANDOM
```
Voting on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model> --seed $RANDOM
```
Few-shot learning, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9> --seed $RANDOM
```
Part segmentation on ShapeNetPart, run:
```
cd segmentation
python main.py --gpu <gpu_id> --ckpts <path/to/pre-trained/model> \
--log_dir <log_dir> --learning_rate 0.0002 --epoch 300 \
--root <path/to/data> \
--seed $RANDOM
```
Semantic segmentation on S3DIS, run:
```
cd semantic_segmentation
python main.py --ckpts <path/to/pre-trained/model> \
--root path/to/data --learning_rate 0.0002 --epoch 60 --gpu <gpu_id> --log_dir <log_dir>
```

## Visualization
Simple visualization, run:
```
python main_vis.py --config cfgs/pretrain/base.yaml --exp_name final_vis \
--ckpts <path/to/pre-trained/model> --test
```

In addition to the simple method mentioned above for visualizing point clouds, we use the [PointFlowRenderer](https://github.com/zekunhao1995/PointFlowRenderer) repository to render high-quality point cloud images.

## Contact

If you have any questions related to the code or the paper, feel free to email Xiangdong (`zhangxiangdong@sjtu.edu.cn`) or Shaofeng (`sherrylone@sjtu.edu.cn`).

## License

PCP-MAE is released under MIT License. See the [LICENSE](./LICENSE) file for more details. Besides, the licensing information for `pointnet2` modules is available [here](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/UNLICENSE).

## Acknowledgements

This codebase is built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [ReCon](https://github.com/qizekun/ReCon), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{zhang2024pcp,
  title={PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders},
  author={Zhang, Xiangdong and Zhang, Shaofeng and Yan, Junchi},
  journal={arXiv preprint arXiv:2408.08753},
  year={2024}
}
```