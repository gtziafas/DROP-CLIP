# DROP-CLIP: 3D Feature Distillation with Object-Centric Priors
This is the official implementation for the paper "3D Feature Distillation with Object-Centric Priors" (arxiv preprint 2025).

<p align="center"> <img src='media/main-figure-1-1.png' align="center" > </p>
Grounding natural language to the physical world is a ubiquitous topic with a wide range of applications in computer vision and robotics. 
Recently, 2D vision-language models such as CLIP have been widely popularized, due to their impressive capabilities for open-vocabulary grounding in 2D images.
Subsequent works aim to elevate 2D CLIP features to 3D via feature distillation, but either learn neural fields that are scene-specific and hence lack generalization, or focus on indoor room scan data that require access to multiple camera views, which is not practical in robot manipulation scenarios.
Additionally, related methods typically fuse features at pixel-level and assume that all camera views are equally informative.
In this work, we show that this approach leads to sub-optimal 3D features, both in terms of grounding accuracy, as well as segmentation crispness.
To alleviate this, we propose a multi-view feature fusion strategy that employs object-centric priors to eliminate uninformative views based on semantic information, and fuse features at object-level via instance segmentation masks. 
To distill our object-centric 3D features, we generate a large-scale synthetic multi-view dataset of cluttered tabletop scenes, spawning 15k scenes from over 3300 unique object instances, which we make publicly available.
We show that our method reconstructs 3D CLIP features with improved grounding capacity and spatial consistency, while doing so from single-view RGB-D, thus departing from the assumption of multiple camera views at test time.
Finally, we show that our approach can generalize to novel tabletop domains and be re-purposed for 3D instance segmentation without fine-tuning, and demonstrate its utility for language-guided robotic grasping in clutter.

[[project page]](https://gtziafas.github.io/DROP_project/) | [[arxiv]](https://arxiv.org/abs/2406.18742) | [[bibtex]](#citation)

## Installation
The code has been tested with `python3.8` with `torch` version 2.0 and CUDA driver 11.7. Create a virtual environment and install `torch` for your own CUDA driver from [here](https://pytorch.org/get-started/locally/). 

Install local dependencies with 
```
pip install -r requirements.txt
```

For installing the `MinkowskiEngine` library, please follow installation instructions from [here](https://github.com/NVIDIA/MinkowskiEngine#anaconda).


## Download MV-TOD dataset
We make the MV-TOD dataset available from [this link](https://drive.google.com/drive/folders/1t25Rvw50vafdeVb2D4J_uJRlnnPe39s-?usp=sharing). You will find the data ready for training under the `processed` folder. Extract all the zip files in your root directory and export it as an environment variable `DATA_ROOT`. You will also find model checkpoints under the `checkpoints` folder. Use the `best_val_miou_model_fine_tune.pth` for the reported results. The raw object assets can be found in the `SyntheticGraspingDataset.zip`.


## Training & Evaluation
We currently support `DistributedDataParallel` training with multiple GPUs. Run:
```
python -m tools/train_distil --config ./config/DistilBlender.yaml
```

Check the config files under `DistilBlender` and modify necessary fields such as `DATA_ROOT`. You can also use the `--opts` flag in the above command to override settings from the config.

To evaluate on the MV-TOD val split, change the `resume` field in the config file to the path to the model checkpoint, then run:
```
python -m tools.validate_blender --config ./config/DistilBlender.yaml
```

For the upper-bound multi-view feature fusion experiments, run:
```
python -m tools.validate_upper_bound --config ./config/DistilBlender.yaml
```

Also check under `scripts` for bash scripts running the other ablation experiments of the paper.

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@misc{tziafas20253dfeaturedistillationobjectcentric,
      title={3D Feature Distillation with Object-Centric Priors}, 
      author={Georgios Tziafas and Yucheng Xu and Zhibin Li and Hamidreza Kasaei},
      year={2025},
      eprint={2406.18742},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.18742}, 
}
```
