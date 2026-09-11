# core/train_base.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from core.data import get_dataloaders
from core.models import BaseClassifier
from utils import save_model, get_logger

logger = get_logger(__name__)

def train_base_classifier():
    device = Config.device
    train_loader, test_loader, classes = get_dataloaders()

    model = BaseClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("开始训练标准 Base Classifier...")
    model.train()
    for epoch in range(15):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        acc = 100 * correct / total
        logger.info(f"Base Epoch {epoch+1:2d} | Loss: {total_loss:.4f} | Acc: {acc:.2f}%")

    save_path = os.path.join(Config.checkpoints_dir, "base_classifier.pth")
    save_model(model, save_path)
    logger.info(f"标准 Base 模型已保存：{save_path}")

if __name__ == "__main__":
    train_base_classifier()