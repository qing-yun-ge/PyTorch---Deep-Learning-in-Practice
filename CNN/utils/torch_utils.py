"""PyTorch工具"""
import torch
import torch.nn as nn


def select_device(device=''):
    """选择设备"""
    if device and 'cuda' in device and torch.cuda.is_available():
        return torch.device(device)
    return torch.device('cpu')


def get_optimizer(model, name='Adam', lr=0.001, weight_decay=0.0001):
    """创建优化器"""
    if name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {name}')


def get_scheduler(optimizer, name='CosineAnnealing', epochs=100, step_size=10, gamma=0.1):
    """创建学习率调度器"""
    if name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    elif name == 'CosineAnnealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return None