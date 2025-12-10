"""验证脚本"""
import argparse
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from models.AlexNet_model import AlexNet
from utils.datasets import create_dataloader
from utils.general import load_yaml, load_checkpoint
from utils.metrics import Metrics
from utils.torch_utils import select_device


def main(opt):
    # 加载配置
    cfg = load_yaml(opt.cfg)
    device = select_device(opt.device or cfg['device'])
    
    print('🔍 Validation')
    
    # 加载模型
    model = AlexNet(cfg['nc'], cfg['dropout']).to(device)
    load_checkpoint(opt.weights, model)
    model.eval()
    
    # 加载数据
    val_loader, val_set = create_dataloader(
        cfg['val'], cfg['img_size'], batch_size=1, 
        shuffle=False, workers=0, augment=False
    )
    print(f'Dataset: {len(val_set)} images')
    
    # 验证
    metrics = Metrics()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            
            metrics.update(preds, labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 结果
    results = metrics.compute()
    print(f'\n📊 Results:')
    print(f'Accuracy:  {results["acc"]:.4f}')
    print(f'Precision: {results["precision"]:.4f}')
    print(f'Recall:    {results["recall"]:.4f}')
    print(f'F1-Score:  {results["f1"]:.4f}')
    
    # 混淆矩阵
    print(f'\n📈 Confusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))
    
    # 分类报告
    print(f'\n📋 Classification Report:')
    print(classification_report(all_labels, all_preds, 
                                target_names=cfg['names'], digits=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/flower.yaml', help='config file')
    parser.add_argument('--weights', type=str, default='runs/train/best.pt', help='model weights')
    parser.add_argument('--device', type=str, default='', help='cuda device')
    opt = parser.parse_args()
    
    main(opt)