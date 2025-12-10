"""评估指标"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Metrics:
    """指标计算器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.preds = []
        self.labels = []
    
    def update(self, pred, label):
        self.preds.extend(pred.cpu().numpy())
        self.labels.extend(label.cpu().numpy())
    
    def compute(self):
        acc = accuracy_score(self.labels, self.preds)
        p, r, f1, _ = precision_recall_fscore_support(
            self.labels, self.preds, average='macro', zero_division=0
        )
        return {'acc': acc, 'precision': p, 'recall': r, 'f1': f1}


class AverageMeter:
    """平均值计算器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count