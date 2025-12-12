import os
import torch
import pandas as pd
from PIL import Image
import argparse
from model import FireClassifier
from dataset import val_transform

# 测试模型并输出结果
def test_model(model, test_dir, output_csv):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 获取所有测试图像
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    results = []
    
    with torch.no_grad():
        for image_name in test_images:
            image_path = os.path.join(test_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image = val_transform(image).unsqueeze(0).to(device)
            
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            results.append((image_name, predicted.item()))
    
    # 保存结果到csv
    df = pd.DataFrame(results, columns=['ID', 'Label'])
    df.to_csv(output_csv, index=False)
    print(f"测试结果保存到{output_csv}")

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Fire Classification Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='checkpoint路径')
    parser.add_argument('--output_csv', type=str, default='test_results.csv', help='测试结果输出路径')
    parser.add_argument('--test_dir', type=str, default='./data/test', help='测试数据目录')
    args = parser.parse_args()
    
    # 加载模型
    model = FireClassifier()
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"从{args.checkpoint}加载模型成功")
    else:
        print(f"找不到checkpoint文件: {args.checkpoint}")
        return
    
    # 测试模型
    print("开始测试模型...")
    test_model(model, args.test_dir, args.output_csv)
    
    print("测试完成!")

if __name__ == '__main__':
    main()