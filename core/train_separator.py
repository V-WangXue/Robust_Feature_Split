# core/train_separator.py
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from .models import SimpleUNet, BaseClassifier, PerceptualLoss
from .data import get_dataloaders
from config import Config
from utils import save_model, create_dirs, get_logger
# ① 新增 SSIM
from pytorch_msssim import ssim

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_image_range(x):
    """将 ImageNet 标准化张量转换回 SSIM 需要的 [0, 1] 范围。"""
    mean = _MEAN.to(device=x.device, dtype=x.dtype)
    std = _STD.to(device=x.device, dtype=x.dtype)
    return ((x * std) + mean).clamp(0, 1)

logger = get_logger(__name__)


def train_feature_separator():
    """训练特征分离器（自编码器）"""
    # 设备设置
    device = Config.device
    logger.info(f"使用设备训练特征分离器: {device}")

    # 加载数据
    train_loader, test_loader, classes = get_dataloaders()

    # 初始化模型
    autoencoder = SimpleUNet().to(device)
    base_classifier = BaseClassifier().to(device)  # 基础分类器
    perceptual_loss = PerceptualLoss().to(device)  # 感知损失模块

    # 定义损失函数和优化器
    criterion_ce = nn.CrossEntropyLoss()  # 分类损失
    criterion_mse = nn.MSELoss()          # MSE损失
    # ② SSIM 权重（可调）
    ssim_weight = 0.5

    # 联合优化自编码器和基础分类器
    optimizer = optim.Adam(
        list(autoencoder.parameters()) + list(base_classifier.parameters()),
        lr=Config.lr_separator,
        weight_decay=Config.weight_decay_sep
    )

    # 训练自编码器（特征分离器）
    logger.info("开始训练特征分离器（自编码器）...")
    start_time = time.time()
    autoencoder.train()
    base_classifier.train()

    for epoch in range(Config.separator_epochs):
        running_loss = 0.0
        running_loss1 = 0.0  # 分类损失
        running_loss2 = 0.0  # 感知损失
        running_loss3 = 0.0  # 特征完整性损失
        running_loss4 = 0.0  # ③ SSIM 损失

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            robust_feat, non_robust_feat = autoencoder(inputs)  # 分离特征
            outputs = base_classifier(robust_feat)              # 保证鲁棒特征保留类别语义

            # 计算各部分损失
            loss1 = criterion_ce(outputs, labels)               # 分类损失
            loss2 = perceptual_loss(inputs, robust_feat)        # 感知一致性损失
            loss3 = criterion_mse(inputs, robust_feat)          # 特征完整性损失
            loss4 = 1 - ssim(
                _to_image_range(inputs), _to_image_range(robust_feat), data_range=1.0,
                size_average=True
            )  # ④ SSIM 损失（骨架不能太糊）

            # 总损失（带权重）
            total_loss = (
                Config.loss1_weight * loss1 +
                Config.loss2_weight * loss2 +
                Config.loss3_weight * loss3 +
                ssim_weight * loss4
            )

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 累计损失
            running_loss += total_loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss3 += loss3.item()
            running_loss4 += loss4.item()

        # 计算平均损失
        avg_loss = running_loss / len(train_loader)
        avg_loss1 = running_loss1 / len(train_loader)
        avg_loss2 = running_loss2 / len(train_loader)
        avg_loss3 = running_loss3 / len(train_loader)
        avg_loss4 = running_loss4 / len(train_loader)

        # 打印训练进度（含 SSIM）
        logger.info(
            f"特征分离器 Epoch {epoch+1}/{Config.separator_epochs} - "
            f"总损失: {avg_loss:.4f}, "
            f"分类损失: {avg_loss1:.4f}, "
            f"感知损失: {avg_loss2:.4f}, "
            f"特征损失: {avg_loss3:.4f}, "
            f"SSIM损失: {avg_loss4:.4f}"
        )

    # 训练时间统计
    train_time = time.time() - start_time
    logger.info(f"特征分离器训练完成，耗时: {train_time:.2f}秒 ({train_time/60:.1f}分钟)")

    # 保存模型
    create_dirs([Config.checkpoints_dir])
    autoencoder_path = os.path.join(Config.checkpoints_dir, "robust_feature_autoencoder.pth")
    save_model(autoencoder, autoencoder_path)

    classifier_path = os.path.join(Config.checkpoints_dir, "separator_aux_classifier.pth")
    save_model(base_classifier, classifier_path)

    return autoencoder, base_classifier
