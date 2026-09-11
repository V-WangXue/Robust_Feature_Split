import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import random
from config import Config
from utils import get_logger

# 初始化日志
logger = get_logger(__name__)

def get_transforms():
    """获取数据预处理管道"""
    return transforms.Compose([
        transforms.Resize((32, 32)),  # 统一调整为32x32大小
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(10),   # 随机旋转±10度
        transforms.ToTensor(),        # 转换为Tensor
        transforms.Normalize(         # 标准化
            mean=[0.485, 0.456, 0.406],  # 通用RGB均值
            std=[0.229, 0.224, 0.225]    # 通用RGB标准差
        )
    ])

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
    transform = get_transforms()
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(
        root=train_root,
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        root=test_root,
        transform=transform
    )
    
    # 记录类别信息
    classes = train_dataset.classes
    logger.info(f"数据集类别: {classes}")
    logger.info(f"原始训练集大小: {len(train_dataset)}, 原始测试集大小: {len(test_dataset)}")
    
    # 随机采样部分样本（加速训练）
    if Config.train_samples < len(train_dataset):
        train_indices = random.sample(range(len(train_dataset)), Config.train_samples)
        train_dataset = Subset(train_dataset, train_indices)
        logger.info(f"使用训练样本数: {Config.train_samples}")
    
    if Config.test_samples < len(test_dataset):
        test_indices = random.sample(range(len(test_dataset)), Config.test_samples)
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
