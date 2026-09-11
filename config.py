import torch

class Config:
    # 数据集参数
    num_classes = 6  
    train_samples = 3000  
    test_samples = 650  
    random_seed = 42
    
    # 路径设置
    data_dir = "D:/intel-image-classification"  
    train_data_subdir = "seg_train/seg_train"    
    test_data_subdir = "seg_test/seg_test"       
    checkpoints_dir = "./checkpoints"            
    logs_dir = "./logs"                          
    visualizations_dir = "./visualizations"      
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据加载参数
    batch_size = 128  # 批次大小（根据GPU显存调整）
    num_workers = 4   # 数据加载线程数
    
    # 特征分离器训练参数
    separator_epochs = 30    # 自编码器训练轮次
    lr_separator = 5e-4      # 学习率
    weight_decay_sep = 1e-5  # 权重衰减
    
    # 鲁棒分类器训练参数
    base_epochs = 15         # 基础分类器训练轮次
    lr_base = 1e-3           # 基础分类器学习率
    robust_epochs = 60       # 分类器训练轮次
    lr_robust = 0.1         # 学习率
    lr_decay_epoch = 40       # 学习率衰减轮次
    momentum_robust = 0.9    # 动量参数
    weight_decay_rob = 1e-4  # 权重衰减
    
    # 损失函数权重
    loss1_weight = 1.0  # 分类损失权重
    loss2_weight = 1.0  # 感知损失权重
    loss3_weight = 0.5  # 特征完整性损失权重
    
    # 对抗攻击参数
    epsilon_fgsm = 0.05  # FGSM攻击强度
    
    @classmethod
    def validate(cls):
        """验证配置参数有效性"""
        assert cls.batch_size > 0, "批次大小必须为正数"
        assert cls.train_samples > 0 and cls.test_samples > 0, "采样数量必须为正数"
        assert cls.base_epochs > 0 and cls.separator_epochs > 0 and cls.robust_epochs > 0, "训练轮次必须为正数"
        assert cls.num_workers >= 0, "数据加载线程数不能为负数"
        assert cls.lr_base > 0 and cls.lr_separator > 0 and cls.lr_robust > 0, "学习率必须为正数"
        assert 0 < cls.lr_decay_epoch < cls.robust_epochs, "学习率衰减轮次必须位于训练轮次范围内"
        assert 0 < cls.epsilon_fgsm <= 1, "FGSM攻击强度必须在(0, 1]范围内"
        assert cls.num_classes == 6, "Intel数据集固定为6个类别"
