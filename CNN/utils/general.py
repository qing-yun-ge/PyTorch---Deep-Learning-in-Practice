"""通用工具函数"""
import yaml
import torch
from pathlib import Path


def load_yaml(path):
    """加载YAML配置"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_checkpoint(state, path, is_best=False):
    """保存检查点"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = Path(path).parent / 'best.pt'
        torch.save(state, best_path)
        print(f'✅ Best model saved: {best_path}')


def load_checkpoint(path, model, optimizer=None):
    """加载检查点"""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0)