# 文件结构

```
CVPJ/
├── result/
│   ├── checkpoint.pth      # 训练好的模型检查点文件
│   └── test_results.csv    # 测试集预测结果文件
├── dataset.py              # 数据加载与预处理
├── model.py                # 模型结构定义
├── train.py                # 模型训练脚本
├── inference.py            # 使用训练好的模型对测试集进行预测并生成结果文件
├── utils.py                # 工具函数，如训练函数、数据可视化等
├── requirements.txt        # 项目依赖库
└── README.md               # 项目说明文档
```

# 环境配置

Python 3.12.3
CUDA 12.8
其余依赖记录于`requirements.txt`文件中。

# 数据准备方式

训练集使用Kaggle竞赛提供的三个数据集[1](https://nextcloud.ig.umons.ac.be/s/nASDCFgsApgz8iB/download/FIRE_DATABASE_1.zip) [2](https://nextcloud.ig.umons.ac.be/s/pB94TdAMJrr7JtQ/download/FIRE_DATABASE_2.zip) [3](https://nextcloud.ig.umons.ac.be/s/GALdsKZfqDf9TyD/download/FIRE_DATABASE_3.zip)合并构成，完全同名文件仅保留一份（复制并替换），验证集使用Kaggle竞赛提供的数据集[4](https://nextcloud.ig.umons.ac.be/s/GTf7wcpCFraPzxB/download/test.zip)。默认情况下，数据集文件存放于当前目录下的`data`文件夹中，整体结构如下：

```
CVPJ/
├── data/
│   ├── train/      #由三个数据集合并而成
│   │   ├──fire/
│   │   ├──no_fire/
│   │   └──start_fire/
│   ├── val/
│   │   ├──fire/
│   │   ├──no_fire/
│   │   └──start_fire/
│   └── test/
│...
```
如果需要使用额外的训练/验证数据集，应当按照`fire`、`no_fire`、`start_fire`三个文件夹的结构组织，训练时使用`--train_dir`参数指定训练集路径，`--val_dir`参数指定验证集路径。

# 训练步骤

得到最终结果所使用的训练参数均已作为默认参数保存于`train.py`文件中，如果使用相同参数训练，直接运行`train.py`即可。

```bash
python train.py
```

`train.py`文件中可修改的参数如下：

```bash
--num_epochs    # 训练轮数
--batch_size    # 批次大小
--lr            # 学习率
--weight_decay  # 权重衰减
--checkpoint    # 模型检查点保存位置
--loss_history  # 损失函数历史记录保存位置
--loss_plot     # 损失函数历史记录可视化保存位置
--acc_plot      # 准确率历史记录可视化保存位置
--train_dir     # 训练集路径
--val_dir       # 验证集路径
--save_latest   # 是否保存训练完全部轮次后的模型检查点
--latest_path   # 保存训练完全部轮次后的模型检查点路径
```
默认情况下，将会保存训练过程中验证集上损失最低的检查点到当前目录下的`checkpoint.pth`文件，同时也会保存训练完全部轮次后的模型检查点到`checkpoint_latest.pth`文件。为方便从检查点继续训练时的数据可视化，记录了损失函数历史和准确率历史。

# 推理与生成结果文件

```bash
python inference.py
```

默认情况下，将会使用当前目录下的`checkpoint.pth`文件进行推理，将测试集预测结果保存到当前目录下的`test_results.csv`文件中。如果需要使用其他模型检查点进行推理，应当使用`--checkpoint`参数指定模型检查点路径。

`inference.py`文件中可修改的参数如下：

```bash
--checkpoint    # 模型检查点路径
--test_dir      # 测试集路径
--output_csv    # 测试集预测结果保存路径
```

`checkpoint.pth`中保存的不只是训练好的模型权重，而是字典结构，结构如下：

```Python
checkpoint = {
    'epoch': epoch,                                 # 训练轮数
    'model_state_dict': model.state_dict(),         # 模型权重
    'optimizer_state_dict': optimizer.state_dict(), # 优化器状态
    'train_loss': epoch_train_loss,                 # 训练损失
    'train_acc': epoch_train_acc,                   # 训练准确率
    'val_loss': epoch_val_loss,                     # 验证损失
    'val_acc': epoch_val_acc,                       # 验证准确率
    'learning_rate': lr                             # 学习率
}
```

如果不使用`inference.py`文件进行推理，也可以使用如下代码自行加载模型检查点进行推理：

```Python
import torch
from model import FireClassifier

# 加载模型检查点
checkpoint = torch.load('checkpoint.pth')

# 实例化模型
model = FireClassifier()

# 加载模型权重
model.load_state_dict(checkpoint['model_state_dict'])
```