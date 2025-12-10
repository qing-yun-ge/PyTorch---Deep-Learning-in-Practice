"""AlexNet模型"""
import torch.nn as nn
from models.common import ConvBNReLU, LinearBNReLU


class AlexNet(nn.Module):
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
        
        # 特征提取
        self.features = nn.Sequential(
            ConvBNReLU(3, 96, 11, 4, 2),
            nn.MaxPool2d(3, 2),
            ConvBNReLU(96, 256, 5, 1, 2),
            nn.MaxPool2d(3, 2),
            ConvBNReLU(256, 384, 3, 1, 1),
            ConvBNReLU(384, 384, 3, 1, 1),
            ConvBNReLU(384, 256, 3, 1, 1),
            nn.MaxPool2d(3, 2),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            LinearBNReLU(256 * 6 * 6, 2048),
            nn.Dropout(dropout),
            LinearBNReLU(2048, 1024),
            nn.Linear(1024, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x