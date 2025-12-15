import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model import FireClassifier
from dataset import get_data_loaders
from utils import train_model, plot_loss, plot_accuracy

# 设置随机种子，确保实验可复现
torch.manual_seed(42)

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Fire Classification Model')
    parser.add_argument('--num_epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减系数')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='checkpoint保存路径')
    parser.add_argument('--loss_history', type=str, default='loss_history.npy', help='loss历史保存路径')
    parser.add_argument('--loss_plot', type=str, default='loss_curve.png', help='loss曲线保存路径')
    parser.add_argument('--acc_plot', type=str, default='accuracy_curve.png', help='准确率曲线保存路径')
    parser.add_argument('--train_dir', type=str, default='./data/train', help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default='./data/val', help='验证数据目录')
    parser.add_argument('--save_latest', type=bool, default=True, help='是否在训练完成后保存模型到另一个位置')
    parser.add_argument('--latest_path', type=str, default='checkpoint_latest.pth', help='训练完成后模型的保存路径')
    args = parser.parse_args()
    
    # 获取数据加载器
    train_loader, val_loader, class_to_idx = get_data_loaders(
        args.train_dir, 
        args.val_dir, 
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # 初始化模型、损失函数和优化器
    model = FireClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 训练模型
    print("开始训练模型...")
    model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        lr=args.lr,  # 传递学习率参数
        num_epochs=args.num_epochs, 
        checkpoint_path=args.checkpoint, 
        loss_history_path=args.loss_history
    )
    
    # 绘制loss曲线
    print("绘制loss曲线...")
    plot_loss(train_loss, val_loss, save_path=args.loss_plot)
    
    # 绘制准确率曲线
    print("绘制准确率曲线...")
    plot_accuracy(train_acc, val_acc, save_path=args.acc_plot)
    
    # 训练完成后保存模型到指定路径
    if args.save_latest:
        # 将模型移动到CPU以保存
        model = model.to('cpu')
        # 保存模型状态
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': args.num_epochs,
            'train_loss': train_loss[-1] if train_loss else 0,
            'val_loss': val_loss[-1] if val_loss else 0,
            'train_acc': train_acc[-1] if train_acc else 0,
            'val_acc': val_acc[-1] if val_acc else 0,
            'learning_rate': args.lr
        }, args.latest_path)
        print(f"训练完成后模型已保存到{args.latest_path}")
    
    print("训练完成!")

if __name__ == '__main__':
    main()