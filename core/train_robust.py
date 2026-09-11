import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
import torch.nn.functional as F  # 新增：PGD 需要
from torch.utils.data import TensorDataset, DataLoader
from .models import SimpleUNet, BaseClassifier
from .data import get_dataloaders
from config import Config
from utils import save_model, load_model, get_logger, visualize_features

logger = get_logger(__name__)


def _normalized_bounds(x):
    """返回 ImageNet 标准化空间中的像素上下界和标准差。"""
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return -mean / std, (1 - mean) / std, std


def fgsm_attack(model, x, y, eps):
    """在原始像素尺度上执行 FGSM，并返回标准化后的对抗样本。"""
    model.eval()
    lower, upper, std = _normalized_bounds(x)
    eps_normalized = eps / std
    x_adv = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + eps_normalized * grad.sign()
    return torch.maximum(torch.minimum(x_adv, upper), lower)


def generate_robust_features(autoencoder, dataloader):
    """生成鲁棒特征数据集"""
    device = Config.device
    autoencoder.eval()
    robust_features = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            robust_feat, _ = autoencoder(inputs)
            robust_features.append(robust_feat.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(robust_features), np.concatenate(labels_list)


# ==================== PGD-10 攻击 ====================
def pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, iters=10):
    """在原始像素尺度上执行 PGD，返回标准化后的对抗样本。"""
    model.eval()
    lower, upper, std = _normalized_bounds(x)
    eps_normalized = eps / std
    alpha_normalized = alpha / std
    random_delta = (torch.rand_like(x) * 2 - 1) * eps_normalized
    x_adv = x.clone().detach() + random_delta
    x_adv = torch.maximum(torch.minimum(x_adv, upper), lower).requires_grad_(True)

    for _ in range(iters):
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha_normalized * grad.sign()
        x_adv = torch.maximum(
            torch.minimum(x_adv, x + eps_normalized), x - eps_normalized
        )  # 投影回 ε 球
        x_adv = torch.maximum(torch.minimum(x_adv, upper), lower).requires_grad_(True)
    return x_adv.detach()


def train_robust_classifier():
    # 准备工作
    device = Config.device
    train_loader, test_loader, classes = get_dataloaders()

    # 加载预训练的自编码器
    autoencoder = SimpleUNet().to(device)
    autoencoder_path = os.path.join(Config.checkpoints_dir, "robust_feature_autoencoder.pth")
    autoencoder = load_model(autoencoder, autoencoder_path)

    # 生成鲁棒特征数据集
    logger.info("开始生成鲁棒特征...")
    train_robust_feats, train_labels = generate_robust_features(autoencoder, train_loader)
    test_robust_feats, test_labels = generate_robust_features(autoencoder, test_loader)
    logger.info(
        f"鲁棒特征生成完成 - "
        f"训练集: {train_robust_feats.shape}, "
        f"测试集: {test_robust_feats.shape}"
    )

    # 转换为TensorDataset
    train_dataset = TensorDataset(
        torch.from_numpy(train_robust_feats),
        torch.from_numpy(train_labels)
    )
    train_loader_robust = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True
    )

    # 初始化鲁棒分类器
    robust_classifier = BaseClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        robust_classifier.parameters(),
        lr=Config.lr_robust,
        momentum=Config.momentum_robust,
        weight_decay=Config.weight_decay_rob
    )

    # 训练鲁棒分类器
    logger.info("开始训练鲁棒分类器...")
    start_time = time.time()

    for epoch in range(Config.robust_epochs):
        robust_classifier.train()
        running_loss = 0.0

        # 在配置的轮次将学习率降为 1/10
        if epoch == Config.lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = Config.lr_robust / 10
            logger.info(f"学习率调整为: {Config.lr_robust / 10}")

        for inputs, labels in train_loader_robust:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = robust_classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 打印每轮损失
        avg_loss = running_loss / len(train_loader_robust)
        logger.info(
            f"鲁棒分类器 Epoch {epoch+1}/{Config.robust_epochs}, "
            f"平均损失: {avg_loss:.4f}"
        )

    # 训练时间统计
    train_time = time.time() - start_time
    logger.info(f"鲁棒分类器训练完成，耗时: {train_time:.2f}秒 ({train_time/60:.1f}分钟)")

    # 保存鲁棒分类器
    save_path = os.path.join(Config.checkpoints_dir, "robust_classifier.pth")
    save_model(robust_classifier, save_path)

    # 可视化部分样本的特征
    visualize_features(autoencoder, test_loader, classes, Config.visualizations_dir)

    # 评估鲁棒性（FGSM + PGD-10）
    evaluate_robustness(robust_classifier, autoencoder, test_loader, classes)


def evaluate_robustness(robust_classifier, autoencoder, test_loader, classes):
    """评估基础分类器和鲁棒分类器的鲁棒性（FGSM + PGD-10）"""
    device = Config.device
    # 加载基础分类器（用于对比）
    base_classifier = BaseClassifier().to(device)
    base_classifier = load_model(
        base_classifier,
        os.path.join(Config.checkpoints_dir, "base_classifier.pth")
    )
    base_classifier.eval()
    robust_classifier.eval()
    autoencoder.eval()

    class RobustPipeline(nn.Module):
        def __init__(self, separator, classifier):
            super().__init__()
            self.separator = separator
            self.classifier = classifier

        def forward(self, x):
            robust_feat, _ = self.separator(x)
            return self.classifier(robust_feat)

    robust_pipeline = RobustPipeline(autoencoder, robust_classifier).to(device).eval()

    # 评估指标初始化
    total = 0
    correct_base = 0  # 基础分类器-正常样本
    correct_robust = 0  # 鲁棒分类器-正常样本
    correct_base_adv = 0  # 基础分类器-对抗样本(FGSM)
    correct_robust_adv = 0  # 鲁棒分类器-对抗样本(FGSM)
    correct_base_pgd = 0  # 基础分类器-PGD-10
    correct_robust_pgd = 0  # 鲁棒分类器-PGD-10

    # FGSM 参数
    eps_fgsm = Config.epsilon_fgsm

    # PGD 参数
    pgd_eps = 8 / 255
    pgd_alpha = 2 / 255
    pgd_iters = 10

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        total += labels.size(0)

        # 1. 正常样本上的准确率 —— 无梯度推理
        with torch.no_grad():
            outputs_base = base_classifier(inputs)
            _, predicted_base = torch.max(outputs_base.data, 1)
            correct_base += (predicted_base == labels).sum().item()

            outputs_robust = robust_pipeline(inputs)
            _, predicted_robust = torch.max(outputs_robust.data, 1)
            correct_robust += (predicted_robust == labels).sum().item()

        # 2. FGSM 对抗样本 —— 需要梯度段
        base_fgsm = fgsm_attack(base_classifier, inputs, labels, eps_fgsm)
        robust_fgsm = fgsm_attack(robust_pipeline, inputs, labels, eps_fgsm)

        with torch.no_grad():
            outputs_adv = base_classifier(base_fgsm)
            _, predicted_adv = torch.max(outputs_adv.data, 1)
            correct_base_adv += (predicted_adv == labels).sum().item()

            outputs_robust_adv = robust_pipeline(robust_fgsm)
            _, predicted_robust_adv = torch.max(outputs_robust_adv.data, 1)
            correct_robust_adv += (predicted_robust_adv == labels).sum().item()

        # 3. PGD-10 对抗样本 —— 需要梯度段
        base_pgd = pgd_attack(base_classifier, inputs, labels, pgd_eps, pgd_alpha, pgd_iters)
        robust_pgd = pgd_attack(robust_pipeline, inputs, labels, pgd_eps, pgd_alpha, pgd_iters)

        with torch.no_grad():
            outputs_base_pgd = base_classifier(base_pgd)
            _, predicted_base_pgd = torch.max(outputs_base_pgd.data, 1)
            correct_base_pgd += (predicted_base_pgd == labels).sum().item()

            outputs_robust_pgd = robust_pipeline(robust_pgd)
            _, predicted_robust_pgd = torch.max(outputs_robust_pgd.data, 1)
            correct_robust_pgd += (predicted_robust_pgd == labels).sum().item()

    # 计算准确率并打印
    logger.info("\n===== 鲁棒性评估结果 =====")
    logger.info(f"基础分类器 - 正常样本准确率: {100 * correct_base / total:.2f}%")
    logger.info(f"鲁棒分类器 - 正常样本准确率: {100 * correct_robust / total:.2f}%")
    logger.info(f"基础分类器 - FGSM 对抗样本准确率: {100 * correct_base_adv / total:.2f}%")
    logger.info(f"鲁棒分类器 - FGSM 对抗样本准确率: {100 * correct_robust_adv / total:.2f}%")
    logger.info(f"基础分类器 - PGD-10 准确率: {100 * correct_base_pgd / total:.2f}%")
    logger.info(f"鲁棒分类器 - PGD-10 准确率: {100 * correct_robust_pgd / total:.2f}%")
    logger.info("===========================")
