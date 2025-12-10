"""训练脚本"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from models.AlexNet_model import AlexNet
from utils.datasets import create_dataloader
from utils.general import load_yaml, save_checkpoint
from utils.metrics import Metrics, AverageMeter
from utils.torch_utils import select_device, get_optimizer, get_scheduler


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    loss_meter = AverageMeter()
    metrics = Metrics()
    
    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), imgs.size(0))
        metrics.update(outputs.argmax(1), labels)
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    results = metrics.compute()
    results['loss'] = loss_meter.avg
    return results


def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    loss_meter = AverageMeter()
    metrics = Metrics()
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validating'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss_meter.update(loss.item(), imgs.size(0))
            metrics.update(outputs.argmax(1), labels)
    
    results = metrics.compute()
    results['loss'] = loss_meter.avg
    return results


def main(opt):
    # 加载配置
    cfg = load_yaml(opt.cfg)
    device = select_device(opt.device or cfg['device'])
    save_dir = Path(opt.save_dir or cfg['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'🌻 Training on {device}')
    
    # 数据加载
    train_loader, train_set = create_dataloader(
        cfg['train'], cfg['img_size'], cfg['batch_size'], 
        True, cfg['workers'], augment=True
    )
    val_loader, val_set = create_dataloader(
        cfg['val'], cfg['img_size'], cfg['batch_size'], 
        False, cfg['workers'], augment=False
    )
    print(f'Train: {len(train_set)}, Val: {len(val_set)}')
    
    # 模型
    model = AlexNet(cfg['nc'], cfg['dropout']).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(model, cfg['optimizer'], cfg['lr0'], cfg['weight_decay'])
    scheduler = get_scheduler(optimizer, cfg['scheduler'], cfg['epochs'], 
                             cfg.get('step_size', 10), cfg.get('gamma', 0.1))
    
    # 训练
    best_acc = 0
    for epoch in range(1, cfg['epochs'] + 1):
        print(f'\n📊 Epoch {epoch}/{cfg["epochs"]}')
        
        train_results = train_epoch(model, train_loader, criterion, optimizer, device)
        val_results = validate(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step()
        
        print(f'Train - Loss: {train_results["loss"]:.4f}, Acc: {train_results["acc"]:.4f}')
        print(f'Val   - Loss: {val_results["loss"]:.4f}, Acc: {val_results["acc"]:.4f}')
        
        # 保存
        is_best = val_results['acc'] > best_acc
        if is_best:
            best_acc = val_results['acc']
        
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }, save_dir / 'last.pt', is_best)
    
    print(f'\n🎉 Training complete! Best Acc: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/flower.yaml', help='config file')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--save-dir', type=str, default='', help='save directory')
    opt = parser.parse_args()
    
    main(opt)