import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import random
from config import Config
from utils import get_logger

# 初始化日志
logger = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms():
    """获取训练和测试数据预处理管道。测试集不使用随机增强。"""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 统一调整为32x32大小
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(10),   # 随机旋转±10度
        transforms.ToTensor(),        # 转换为Tensor
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    return train_transform, test_transform

def get_dataloaders():
    """获取训练集和测试集的数据加载器"""
    # 构建完整路径
    train_root = os.path.join(Config.data_dir, Config.train_data_subdir)
    test_root = os.path.join(Config.data_dir, Config.test_data_subdir)
    
    # 验证路径有效性
    if not os.path.exists(train_root):
        raise FileNotFoundError(f"训练集路径不存在: {train_root}")
    if not os.path.exists(test_root):
        raise FileNotFoundError(f"测试集路径不存在: {test_root}")
    
    # 获取预处理管道
    train_transform, test_transform = get_transforms()
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(
        root=train_root,
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=test_root,
        transform=test_transform
    )

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            f"训练集与测试集类别不一致: {train_dataset.classes} != {test_dataset.classes}"
        )
    if len(train_dataset.classes) != Config.num_classes:
        raise ValueError(
            f"数据集包含 {len(train_dataset.classes)} 类，但配置为 {Config.num_classes} 类"
        )
    
    # 记录类别信息
    classes = train_dataset.classes
    logger.info(f"数据集类别: {classes}")
    logger.info(f"原始训练集大小: {len(train_dataset)}, 原始测试集大小: {len(test_dataset)}")
    
    # 随机采样部分样本（加速训练）
    if Config.train_samples < len(train_dataset):
        rng = random.Random(Config.random_seed)
        train_indices = rng.sample(range(len(train_dataset)), Config.train_samples)
        train_dataset = Subset(train_dataset, train_indices)
        logger.info(f"使用训练样本数: {Config.train_samples}")
    
    if Config.test_samples < len(test_dataset):
        rng = random.Random(Config.random_seed + 1)
        test_indices = rng.sample(range(len(test_dataset)), Config.test_samples)
        test_dataset = Subset(test_dataset, test_indices)
        logger.info(f"使用测试样本数: {Config.test_samples}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True if Config.device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True if Config.device == "cuda" else False
    )
    
    return train_loader, test_loader, classes
