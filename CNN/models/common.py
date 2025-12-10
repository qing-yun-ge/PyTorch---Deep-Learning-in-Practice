"""通用模块"""
import torch.nn as nn
class ConvBNReLU(nn.Module):
    """卷积 + BN + ReLU"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LinearBNReLU(nn.Module):
    """全连接 + BN + ReLU"""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)
        self.bn = nn.BatchNorm1d(out_f)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.fc(x)))