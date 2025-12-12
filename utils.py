import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, checkpoint_path=None, loss_history_path=None):
    # 加载已有loss历史
    train_loss_history = []
    val_loss_history = []
    if loss_history_path and os.path.exists(loss_history_path):
        loss_data = np.load(loss_history_path, allow_pickle=True).item()
        train_loss_history = loss_data['train_loss']
        val_loss_history = loss_data['val_loss']
    
    # 加载checkpoint
    start_epoch = 0
    # 初始化最小验证损失
    min_val_loss = float('inf')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # 从checkpoint中获取之前的最小验证损失
        min_val_loss = checkpoint['val_loss']
        print(f"从checkpoint继续训练，开始于第{start_epoch}个epoch")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
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
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            # 添加tqdm进度条到验证循环
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # 只在验证损失小于当前最小值时保存checkpoint
        if checkpoint_path and epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"验证损失下降到{min_val_loss:.4f}，保存checkpoint到{checkpoint_path}")
        
        # 保存loss历史
        if loss_history_path:
            loss_data = {
                'train_loss': train_loss_history,
                'val_loss': val_loss_history
            }
            np.save(loss_history_path, loss_data)
            print(f"Loss历史保存到{loss_history_path}")
    
    return model, train_loss_history, val_loss_history

# 绘制loss曲线
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