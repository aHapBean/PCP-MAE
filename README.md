# PCP-MAE

> [**PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders**](https://arxiv.org/abs/2408.08753) **NeurIPS 2024 spotlight** <br>
> [Xiangdong Zhang](https://scholar.google.com/citations?user=5S-TKKoAAAAJ&hl=zh-CN&oi=sra)\*, [Shaofeng Zhang](https://scholar.google.com/citations?user=VoVVJIgAAAAJ&hl=zh-CN&oi=sra)\* and [Junchi Yan](https://scholar.google.com/citations?user=ga230VoAAAAJ&hl=zh-CN&oi=sra) <br>

[Arxiv](https://arxiv.org/abs/2408.08753)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=pcp-mae-learning-to-predict-centers-for-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/few-shot-3d-point-cloud-classification-on-1)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-1?p=pcp-mae-learning-to-predict-centers-for-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/few-shot-3d-point-cloud-classification-on-2)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-2?p=pcp-mae-learning-to-predict-centers-for-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/few-shot-3d-point-cloud-classification-on-3)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-3?p=pcp-mae-learning-to-predict-centers-for-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pcp-mae-learning-to-predict-centers-for-point/few-shot-3d-point-cloud-classification-on-4)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-4?p=pcp-mae-learning-to-predict-centers-for-point)

https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn

## PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders

<div  align="center">    
 <img src="./figs/overview.jpg" width = "1100"  align=center />
</div>

<!-- 这图有点糊 -->

## 📰 News

<!-- - 🍾 Oct, 2024: The corresponding checkpoints are released. -->
<!-- - 📌 Oct, 2024: The training and inference code is released. -->
- 🎉 Sept, 2024: [**PCP-MAE**](https://arxiv.org/abs/2408.08753) is accepted by NeurIPS 2024 as **spotlight**.
- 💥 Aug, 2024: PCP-MAE is available in [arxiv](https://arxiv.org/abs/2408.08753).


## ✅ TODO List
- [ ] Complete the introduction for the PCP-MAE project.
- [ ] Publish the training and inference code.
- [ ] Release the checkpoints for pre-training and finetuning.

## 3. PCP-MAE Models
| Task              | Dataset        | Config                                                               | Acc.       | Checkpoints Download                                                                                     |
|-------------------|----------------|----------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------|
| Pre-training      | ShapeNet       | [base.yaml](cfgs/pretrain/base.yaml)                        | N.A.       | TBD           |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/finetune_scan_objbg.yaml)     | 95.52%     | TBD          |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/finetune_scan_objonly.yaml) | 94.32%     | TBD        |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/finetune_scan_hardest.yaml) | 90.35%     | TBD       |
| Classification    | ModelNet40(1k) w/o voting | [finetune_modelnet.yaml](./cfgs/finetune_modelnet.yaml)         | 94.1%      | TBD     |
| Classification    | ModelNet40(1k) w/ voting | [finetune_modelnet.yaml](./cfgs/finetune_modelnet.yaml)         | 94.2%      | TBD     |
| Part Segmentation | ShapeNetPart   | [segmentation](./segmentation)                                       | 86.9% Cls.mIoU | TBD        |
| Scene Segmentation | S3DIS   | [semantic_segmentataion](./semantic_segmentation)                                       | 61.3% mIoU | TBD        |

| Task              | Dataset    | Config                                   | 5w10s (%)  | 5w20s (%)  | 10w10s (%) | 10w20s (%) | Download                                                                                       |
|-------------------|------------|------------------------------------------|------------|------------|------------|------------|------------------------------------------------------------------------------------------------|
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/fewshot.yaml) | 97.4 ± 2.3 | 99.1 ± 0.8 | 93.5±3.7 | 95.9±2.7 | TBD |


## Visualization
We use [PointFlowRenderer](https://github.com/zekunhao1995/PointFlowRenderer) repo to render beautiful point cloud image.


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