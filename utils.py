import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, checkpoint_path=None, loss_history_path=None):
    # 加载已有loss和准确率历史
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    if loss_history_path and os.path.exists(loss_history_path):
        loss_data = np.load(loss_history_path, allow_pickle=True).item()
        train_loss_history = loss_data['train_loss']
        val_loss_history = loss_data['val_loss']
        if 'train_acc' in loss_data and 'val_acc' in loss_data:
            train_acc_history = loss_data['train_acc']
            val_acc_history = loss_data['val_acc']
    
    # 加载checkpoint
    start_epoch = 0
    # 初始化最小验证损失
    min_val_loss = float('inf')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)  # 添加map_location参数
        model.load_state_dict(checkpoint['model_state_dict'])
        # 先将模型移动到设备
        model.to(device)
        # 然后加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 将优化器的所有参数移动到设备上
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        
        start_epoch = checkpoint['epoch'] + 1
        # 从checkpoint中获取之前的最小验证损失
        min_val_loss = checkpoint['val_loss']
        print(f"从checkpoint继续训练，开始于第{start_epoch}个epoch")
    else:
        # 如果没有checkpoint，直接将模型移动到设备
        model.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 添加tqdm进度条到训练循环
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)
        
        epoch_train_acc = correct / total
        train_acc_history.append(epoch_train_acc)
        
        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            # 添加tqdm进度条到验证循环
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                # 计算验证准确率
                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        
        epoch_val_acc = correct_val / total_val
        val_acc_history.append(epoch_val_acc)
        
        # 打印损失和准确率信息
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # 只在验证损失小于当前最小值时保存checkpoint
        if checkpoint_path and epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'train_acc': epoch_train_acc,
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  验证损失下降到{min_val_loss:.4f}，保存checkpoint到{checkpoint_path}")
        
        # 保存loss和准确率历史
        if loss_history_path:
            loss_data = {
                'train_loss': train_loss_history,
                'val_loss': val_loss_history,
                'train_acc': train_acc_history,
                'val_acc': val_acc_history
            }
            np.save(loss_history_path, loss_data)
            print(f"  Loss和准确率历史保存到{loss_history_path}")
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# 绘制loss和准确率曲线
def plot_loss(train_loss, val_loss, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss曲线保存到{save_path}")
    
    plt.show()

# 绘制准确率曲线
def plot_accuracy(train_acc, val_acc, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train Accuracy')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"准确率曲线保存到{save_path}")
    
    plt.show()