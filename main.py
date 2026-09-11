from core.train_separator import train_feature_separator
from core.train_robust import train_robust_classifier
from config import Config
from utils import create_dirs, get_logger

def main():
    # 初始化日志
    logger = get_logger(__name__)
    logger.info("===== 鲁棒特征分离与分类项目启动 =====")
    
    # 验证配置并创建目录
    Config.validate()
    create_dirs([
        Config.checkpoints_dir,
        Config.logs_dir,
        Config.visualizations_dir
    ])
    
    # 打印配置信息
    logger.info(f"使用设备: {Config.device}")
    logger.info(f"数据集路径: {Config.data_dir}")
    logger.info(f"类别数量: {Config.num_classes}")
    
    # 执行训练流程
    try:
        # 第一步：训练特征分离器（自编码器）
        logger.info("\n===== 开始训练特征分离器 =====")
        train_feature_separator()
        
        # 第二步：训练鲁棒分类器并评估
        logger.info("\n===== 开始训练鲁棒分类器 =====")
        train_robust_classifier()
        
        logger.info("\n===== 项目运行完成 =====")
    except Exception as e:
        logger.error(f"项目运行出错: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
