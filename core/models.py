import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class UNetPlusPlus(nn.Module):
    """UNet++ 骨架，返回 3×32×32 骨架图"""
    def __init__(self, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        filters = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        # 编码
        self.conv0_0 = self._block(3, filters[0])
        self.conv1_0 = self._block(filters[0], filters[1])
        self.conv2_0 = self._block(filters[1], filters[2])
        self.conv3_0 = self._block(filters[2], filters[3])
        self.conv4_0 = self._block(filters[3], filters[4])

        # 解码
        self.conv0_1 = self._block(filters[0]+filters[1], filters[0])
        self.conv1_1 = self._block(filters[1]+filters[2], filters[1])
        self.conv2_1 = self._block(filters[2]+filters[3], filters[2])
        self.conv3_1 = self._block(filters[3]+filters[4], filters[3])

        self.conv0_2 = self._block(filters[0]*2+filters[1], filters[0])
        self.conv1_2 = self._block(filters[1]*2+filters[2], filters[1])
        self.conv2_2 = self._block(filters[2]*2+filters[3], filters[2])

        self.conv0_3 = self._block(filters[0]*3+filters[1], filters[0])
        self.conv1_3 = self._block(filters[1]*3+filters[2], filters[1])

        self.conv0_4 = self._block(filters[0]*4+filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], 3, 1, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 下采样
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))

        # 上采样+跳跃
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, scale_factor=2)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, scale_factor=2)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, scale_factor=2)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, scale_factor=2)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, scale_factor=2)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, scale_factor=2)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, scale_factor=2)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, scale_factor=2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, scale_factor=2)], 1))

        return self.final(x0_4)          # 3×32×32 骨架图


class SimpleUNet(nn.Module):
    """接口保持 100% 兼容：输入 x → 返回 (骨架, 细节)"""
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.unetpp = UNetPlusPlus(deep_supervision=False)

    def forward(self, x):
        robust = self.unetpp(x)
        non_robust = x - robust
        return robust, non_robust


class BaseClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(BaseClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc   = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features[:23])).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        return nn.MSELoss()(self.features(x), self.features(y))