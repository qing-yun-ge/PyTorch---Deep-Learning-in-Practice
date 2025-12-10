"""数据集加载"""
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def create_dataloader(path, img_size=224, batch_size=16, shuffle=True, workers=4, augment=False):
    """
    创建数据加载器
    
    Args:
        path: 数据集路径
        img_size: 图像尺寸
        batch_size: 批次大小
        shuffle: 是否打乱
        workers: 工作线程数
        augment: 是否使用数据增强（True=训练集，False=验证集）
    """
    
    # 训练集数据增强
    if augment:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),      # 随机裁剪并缩放
            transforms.RandomHorizontalFlip(),           # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # 验证集数据增强
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),     # 直接缩放到目标尺寸
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle, num_workers=workers, pin_memory=True)
    return loader, dataset