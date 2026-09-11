# 鲁棒特征分离与分类项目（Intel图像数据集版）

## 项目概述
本项目实现了基于自监督学习的鲁棒特征与非鲁棒特征分离方法，并构建了对对抗攻击具有更强抵抗力的鲁棒分类模型。项目针对`intel-image-classification`数据集（包含6类自然场景图像）进行了专门优化。

## 核心功能
1. 使用简化UNet结构的自编码器分离图像中的鲁棒特征（如场景轮廓）和非鲁棒特征（如细节纹理）
2. 多损失函数协同优化特征分离效果
3. 基于鲁棒特征训练分类器，提升模型对对抗攻击的抵抗能力
4. 提供完整的特征可视化和性能评估功能

## 项目结构robust_feature_project/
├─ core/                 # 核心功能模块
│  ├─ models.py          # 模型架构定义（自编码器、分类器等）
│  ├─ data.py            # 数据加载与预处理
│  ├─ train_separator.py # 特征分离器训练逻辑
│  └─ train_robust.py    # 鲁棒分类器训练与评估
├─ config.py             # 项目配置参数
├─ utils.py              # 通用工具函数
├─ main.py               # 项目主入口
├─ requirements.txt      # 依赖库列表
└─ README.md             # 项目说明文档
## 环境准备
1. 确保已安装Python 3.8-3.10版本
2. 安装依赖库：
   ```bash
   pip install -r requirements.txt
   ```
3. （推荐）配置NVIDIA GPU环境以加速训练：
   - 安装NVIDIA驱动
   - 安装对应版本的CUDA Toolkit
   - 安装支持CUDA的PyTorch版本

## 数据集准备
1. 本项目使用`intel-image-classification`数据集
2. 数据集应包含以下结构：
   ```
   D:/intel-image-classification/
   ├─ seg_train/
   │  └─ seg_train/
   │     ├─ buildings/
   │     ├─ forest/
   │     ├─ glacier/
   │     ├─ mountain/
   │     ├─ sea/
   │     └─ street/
   └─ seg_test/
      └─ seg_test/
         ├─ buildings/
         └─ ...（其他类别）
   ```
3. 可在`config.py`中修改数据集路径

## 快速开始
直接运行主入口文件：python main.py程序将自动执行：
1. 数据加载与预处理
2. 特征分离器（自编码器）训练
3. 鲁棒分类器训练
4. 模型性能与鲁棒性评估

## 配置参数调整
所有可配置参数均在`config.py`中定义，主要包括：
- 数据集路径和样本数量
- 训练参数（批次大小、学习率、训练轮次等）
- 损失函数权重
- 对抗攻击强度

## 输出结果
训练完成后，将生成以下结果：
1. 模型文件（保存在`./checkpoints`）：
   - `robust_feature_autoencoder.pth`：特征分离器模型
   - `base_classifier.pth`：基础分类器
   - `robust_classifier.pth`：鲁棒分类器

2. 可视化结果（保存在`./visualizations`）：
   - 原始图像、鲁棒特征和非鲁棒特征的对比图

3. 评估指标：
   - 基础分类器和鲁棒分类器在正常样本上的准确率
   - 基础分类器和鲁棒分类器在对抗样本上的准确率