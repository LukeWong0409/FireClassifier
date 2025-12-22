import torch
import argparse
from model import FireClassifier
from dataset import get_data_loaders

def evaluate_per_class(model, data_loader, class_names, device):
    """在指定数据集上评估模型，计算每个类别的准确率，并记录正确和错误分类的图像"""
    model.eval()
    correct_per_class = {class_name: 0 for class_name in class_names}
    total_per_class = {class_name: 0 for class_name in class_names}
    
    # 用于存储每个类别的正确和错误分类的图像信息
    correct_images = {class_name: [] for class_name in class_names}
    incorrect_images = {class_name: [] for class_name in class_names}
    
    # 获取数据集的图像路径和标签信息
    dataset = data_loader.dataset
    img_paths = dataset.imgs
    
    with torch.no_grad():
        batch_start_idx = 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # 获取当前批次的图像路径
            batch_img_paths = img_paths[batch_start_idx:batch_start_idx + len(inputs)]
            
            for i in range(len(inputs)):
                label_idx = labels[i].item()
                pred_idx = predicted[i].item()
                class_name = class_names[label_idx]
                pred_class_name = class_names[pred_idx]
                img_path, _ = batch_img_paths[i]
                img_name = img_path.split('\\')[-1]  # 获取图像文件名
                
                total_per_class[class_name] += 1
                
                if label_idx == pred_idx:
                    correct_per_class[class_name] += 1
                    correct_images[class_name].append(img_name)
                else:
                    incorrect_images[class_name].append((img_name, pred_class_name))
            
            batch_start_idx += len(inputs)
    
    # 计算每个类别的准确率
    accuracy_per_class = {}
    for class_name in class_names:
        if total_per_class[class_name] > 0:
            accuracy_per_class[class_name] = correct_per_class[class_name] / total_per_class[class_name]
        else:
            accuracy_per_class[class_name] = 0.0
    
    # 计算总体准确率
    total_correct = sum(correct_per_class.values())
    total_samples = sum(total_per_class.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return accuracy_per_class, overall_accuracy, total_per_class, correct_per_class, correct_images, incorrect_images

def main():
    parser = argparse.ArgumentParser(description='Evaluate Fire Classification Model on Train and Val Sets')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='模型checkpoint路径')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--train_dir', type=str, default='./data/train', help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default='./data/val', help='验证数据目录')
    parser.add_argument('--num_samples', type=int, default=5, help='每个类别输出的样本数量')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    train_loader, val_loader, class_to_idx = get_data_loaders(
        args.train_dir, 
        args.val_dir, 
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # 创建类别名称列表 (按索引排序)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"类别列表: {class_names}")
    
    # 初始化模型
    model = FireClassifier(num_classes=len(class_names))
    
    # 加载checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"模型加载完成，来自epoch {checkpoint['epoch'] + 1}")
    
    print("\n" + "="*50)
    print("评估训练集")
    print("="*50)
    train_acc_per_class, train_overall_acc, train_total_per_class, train_correct_per_class, train_correct_images, train_incorrect_images = evaluate_per_class(
        model, train_loader, class_names, device
    )
    
    # 输出训练集结果
    print(f"训练集总体准确率: {train_overall_acc:.4f}")
    print("\n训练集各类别准确率:")
    for class_name in class_names:
        print(f"  {class_name}: {train_acc_per_class[class_name]:.4f} ({train_acc_per_class[class_name]*100:.2f}%) "
              f"({train_correct_per_class[class_name]} / {train_total_per_class[class_name]})")
    
    # 输出训练集每个类别分类正确的图像
    print("\n" + "="*50)
    print("训练集分类正确的图像示例:")
    print("="*50)
    for class_name in class_names:
        print(f"\n{class_name} 类别分类正确的图像:")
        count = min(args.num_samples, len(train_correct_images[class_name]))
        for i in range(count):
            print(f"  {train_correct_images[class_name][i]}")
    
    # 输出训练集每个类别分类错误的图像
    print("\n" + "="*50)
    print("训练集分类错误的图像示例:")
    print("="*50)
    for class_name in class_names:
        print(f"\n{class_name} 类别分类错误的图像:")
        count = min(args.num_samples, len(train_incorrect_images[class_name]))
        for i in range(count):
            img_name, pred_class = train_incorrect_images[class_name][i]
            print(f"  {img_name} - 被错误分类为: {pred_class}")
    
    print("\n" + "="*50)
    print("评估验证集")
    print("="*50)
    val_acc_per_class, val_overall_acc, val_total_per_class, val_correct_per_class, val_correct_images, val_incorrect_images = evaluate_per_class(
        model, val_loader, class_names, device
    )
    
    # 输出验证集结果
    print(f"验证集总体准确率: {val_overall_acc:.4f}")
    print("\n验证集各类别准确率:")
    for class_name in class_names:
        print(f"  {class_name}: {val_acc_per_class[class_name]:.4f} ({val_acc_per_class[class_name]*100:.2f}%) "
              f"({val_correct_per_class[class_name]} / {val_total_per_class[class_name]})")
    
    # 输出验证集每个类别分类正确的图像
    print("\n" + "="*50)
    print("验证集分类正确的图像示例:")
    print("="*50)
    for class_name in class_names:
        print(f"\n{class_name} 类别分类正确的图像:")
        count = min(args.num_samples, len(val_correct_images[class_name]))
        for i in range(count):
            print(f"  {val_correct_images[class_name][i]}")
    
    # 输出验证集每个类别分类错误的图像
    print("\n" + "="*50)
    print("验证集分类错误的图像示例:")
    print("="*50)
    for class_name in class_names:
        print(f"\n{class_name} 类别分类错误的图像:")
        count = min(args.num_samples, len(val_incorrect_images[class_name]))
        for i in range(count):
            img_name, pred_class = val_incorrect_images[class_name][i]
            print(f"  {img_name} - 被错误分类为: {pred_class}")
    
    # 输出总结表格
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print(f"{'类别':<15} {'训练集准确率':<15} {'验证集准确率':<15} {'训练集样本数':<15} {'验证集样本数':<15}")
    print("-"*70)
    for class_name in class_names:
        train_acc = f"{train_acc_per_class[class_name]*100:.2f}%"
        val_acc = f"{val_acc_per_class[class_name]*100:.2f}%"
        train_count = train_total_per_class[class_name]
        val_count = val_total_per_class[class_name]
        print(f"{class_name:<15} {train_acc:<15} {val_acc:<15} {train_count:<15} {val_count:<15}")
    print("-"*70)
    print(f"{'总体':<15} {train_overall_acc*100:.2f}%       {val_overall_acc*100:.2f}%       {sum(train_total_per_class.values()):<15} {sum(val_total_per_class.values()):<15}")

if __name__ == '__main__':
    main()