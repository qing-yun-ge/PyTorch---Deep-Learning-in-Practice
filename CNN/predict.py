"""预测脚本"""
import argparse
import torch
from PIL import Image
from torchvision import transforms

from models.AlexNet_model import AlexNet
from utils.general import load_yaml, load_checkpoint
from utils.torch_utils import select_device


def main(opt):
    # 加载配置
    cfg = load_yaml(opt.cfg)
    device = select_device(opt.device or cfg['device'])
    
    print('🔮 Prediction')
    
    # 加载模型
    model = AlexNet(cfg['nc'], cfg['dropout']).to(device)
    load_checkpoint(opt.weights, model)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(opt.source).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(1).item()
        confidence = probs[0][pred_class].item()
        
        # Top-K
        top_k_prob, top_k_idx = torch.topk(probs, opt.top_k)
    
    # 结果
    print(f'\n🌸 Prediction:')
    print(f'Class: {cfg["names"][pred_class]}')
    print(f'Confidence: {confidence:.4f} ({confidence*100:.2f}%)')
    
    print(f'\n📊 Top-{opt.top_k}:')
    for i, (prob, idx) in enumerate(zip(top_k_prob[0], top_k_idx[0]), 1):
        name = cfg["names"][idx.item()]
        print(f'{i}. {name:12s} - {prob.item():.4f} ({prob.item()*100:.2f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='image path')
    parser.add_argument('--cfg', type=str, default='configs/flower.yaml', help='config file')
    parser.add_argument('--weights', type=str, default='runs/train/best.pt', help='model weights')
    parser.add_argument('--device', type=str, default='', help='cuda device')
    parser.add_argument('--top-k', type=int, default=3, help='top k predictions')
    opt = parser.parse_args()
    
    main(opt)