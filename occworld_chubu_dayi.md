# OccWorld 项目初步答疑

## 项目依赖：mmdetection3d 和 Occ3D

- **mmdetection3d 的支撑作用**  
  OccWorld 的数据加载、配置组织和部分几何工具直接依托于 mmdetection3d 生态：  
  1. `dataset/__init__.py` 通过 `mmengine.registry.Registry` 构建 `OPENOCC_DATASET` 与 `OPENOCC_DATAWRAPPER`，沿用了 mmdetection3d/mmengine 的模块化注册体系，便于按配置文件动态实例化数据集与封装器。  
  2. `dataset/dataset.py` 中的 `nuScenesSceneDatasetLidar` 类被注册进该 Registry，并调用 `mmdet3d.structures.bbox_3d` 的 `LiDARInstance3DBoxes`、`Box3DMode` 完成 3D 框格式转换，体现了对 mmdetection3d 几何数据结构的直接复用。  
  3. 训练配置（例如 `config/train_occworld.py`）以 mmdetection3d 通用的字典式配置描述优化器、数据集、DataLoader 等组件，使 OccWorld 能够与 OpenMMLab 现有工具链（多卡训练、日志记录、评测脚本等）无缝衔接。
- **Occ3D 的支撑作用**  
  OccWorld 的监督信号来自 3D 语义占据网格，官方推荐直接使用 Occ3D 生成的 `gts/scene/token/labels.npz`：  
  1. `nuScenesSceneDatasetLidar.__getitem__` 会按照时间序列读取这些 `.npz` 文件，将 `semantics` 体素栅格堆叠成模型的输入 `input_occs` 与输出 `output_occs`，训练过程中实时提供语义占据标签。  
  2. Occ3D 定义的体素坐标系、语义类别映射与数据预处理流程在 OccWorld 中原样沿用（如 `config/label_mapping/nuscenes-occ.yaml`），保证模型输出与语义真值一致。  
  3. Occ3D 还提供了生成真值所需的脚本与数据准备流程，免除了 OccWorld 用户自行构建稠密占据标注的成本。

## 数据依赖：nuScenes 原始数据与 PKL 清单的协同

- **原始 nuScenes 数据的作用**  
  `config/train_occworld.py` 中 `data_path = 'data/nuscenes/'`，要求建立到官方 nuScenes 数据集的软链接（含 `samples/`、`sweeps/`、`maps/` 等目录）。这些原始传感器数据用于：  
  1. 供 Occ3D 生成语义占据网格；  
  2. 让模型在推理或可视化阶段可回溯到原始多传感器帧（`dataset/dataset.py` 的 `get_image_info` 会访问相机图像路径、传感器外参）。  
- **Occ3D 生成的 `gts/` 目录**  
  在 `data/nuscenes/gts/scene/token/labels.npz` 中保存每个时刻的语义占据栅格，`nuScenesSceneDatasetLidar` 通过传入参数 `input_dataset='gts'` 与 `output_dataset='gts'` 读取这些网格，分别构造模型输入（历史帧）与监督标签（未来帧）。
- **PKL 清单的具体作用**  
  README 提供的 `nuscenes_infos_train_temporal_v3_scene.pkl` 与 `nuscenes_infos_val_temporal_v3_scene.pkl` 是对原始数据和占据真值的索引扩展，并未包含图像或点云本体：  
  1. `dataset/dataset.py` 首先 `pickle.load(imageset)`，得到 `data['infos']`，其中每个场景按时间顺序列出帧信息（token、传感器外参、未来轨迹、占据标签路径等）。  
  2. 这些信息驱动采样逻辑：`__getitem__` 按场景序列随机抽取连续帧，将历史占据作为输入、未来占据作为监督，同时整理位姿、相机参数等元数据。  
  3. PKL 条目里的路径字段依赖于 `data/nuscenes` 下的原始数据与 `gts/` 目录，因此 PKL 不能独立使用，而是作为“导航索引”加速数据装载与时间序列拼接。

## 数据流简述

1. 准备官方 nuScenes 数据并生成 Occ3D 的语义占据（`gts/`）。  
2. 下载官方提供的 train/val PKL，并放置于 `data/`，用于描述场景序列与每帧的关键信息。  
3. 训练配置（如 `config/train_occworld.py`）通过 Registry 构建 `nuScenesSceneDatasetLidar`，按照 PKL 描述加载 nuScenes 原始传感器数据与 Occ3D 占据真值。  
4. DataLoader 将历史占据栅格送入 OccWorld 的 VQ-VAE + Transformer 结构，预测未来语义占据及自车运动。  
5. 训练/评估过程中依赖 mmdetection3d 提供的几何工具和 mmengine/mmdet 风格的配置、采样与并行基础设施。
