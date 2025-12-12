import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 获取数据加载器
def get_data_loaders(train_dir, val_dir, batch_size=32, num_workers=0):
    # 检查训练数据是否包含所有三个类别
    train_classes = os.listdir(train_dir)
    expected_classes = ['fire', 'no_fire', 'start_fire']
    missing_classes = [cls for cls in expected_classes if cls not in train_classes]
    
    print(f"训练数据包含的类别: {train_classes}")
    if missing_classes:
        print(f"警告: 训练数据缺少以下类别: {missing_classes}")
    
    # 数据加载器
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"类别映射: {train_dataset.class_to_idx}")
    
    return train_loader, val_loader, train_dataset.class_to_idx