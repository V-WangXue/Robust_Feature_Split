import os
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from config import Config

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'      # 用系统默认无衬线字体
matplotlib.rcParams['axes.unicode_minus'] = False          # 解决负号显示方块

def create_dirs(dirs):
    """创建目录列表中的所有目录"""
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

def save_model(model, path):
    """保存模型权重"""
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    torch.save(model.state_dict(), path)
    get_logger(__name__).info(f"模型已保存至: {path}")

def load_model(model, path):
    """加载模型权重"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        state_dict = torch.load(path, map_location=Config.device, weights_only=True)
        model.load_state_dict(state_dict)
        get_logger(__name__).info(f"模型已加载: {path}")
        return model
    except Exception as e:
        get_logger(__name__).error(f"模型加载失败: {str(e)}", exc_info=True)
        raise

def get_logger(name):
    """获取日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件输出
        create_dirs([Config.logs_dir])
        file_handler = logging.FileHandler(os.path.join(Config.logs_dir, "train.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def visualize_features(autoencoder, dataloader, classes, save_dir, num_samples=5):
    """可视化原始图像、鲁棒特征和非鲁棒特征"""
    device = Config.device
    autoencoder.eval()
    samples_visualized = 0
    
    # 创建保存目录
    create_dirs([save_dir])
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if samples_visualized >= num_samples:
                break
                
            inputs = inputs.to(device)
            robust_feat, non_robust_feat = autoencoder(inputs)
            
            # 转换为CPU并反标准化
            inputs_np = inputs.cpu().numpy()
            robust_np = robust_feat.cpu().numpy()
            non_robust_np = non_robust_feat.cpu().numpy()
            labels_np = labels.numpy()
            
            # 遍历批次中的样本
            for i in range(inputs_np.shape[0]):
                if samples_visualized >= num_samples:
                    break
                    
                # 准备图像数据（调整维度和标准化）
                img = inputs_np[i].transpose(1, 2, 0)
                robust = robust_np[i].transpose(1, 2, 0)
                non_robust = non_robust_np[i].transpose(1, 2, 0)
                
                # 反标准化（使用数据加载时的均值和标准差）
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                robust = std * robust + mean
                non_robust = std * non_robust + mean
                
                # 确保像素值在[0,1]范围内
                img = np.clip(img, 0, 1)
                robust = np.clip(robust, 0, 1)
                non_robust = np.clip(non_robust, 0, 1)
                
                # 创建可视化图像
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img)
                axes[0].set_title(f"Original: {classes[labels_np[i]]}")  # 原始图像
                axes[0].axis('off')
                
                axes[1].imshow(robust)
                axes[1].set_title("Robust Feature") # 鲁棒特征
                axes[1].axis('off')
                
                axes[2].imshow(non_robust)
                axes[2].set_title("Non-Robust Feature") # 非鲁棒特征
                axes[2].axis('off')
                
                # 保存图像
                save_path = os.path.join(save_dir, f"feature_visualization_{samples_visualized}.png")
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                samples_visualized += 1
    
    get_logger(__name__).info(f"特征可视化完成，已保存至: {save_dir}")
