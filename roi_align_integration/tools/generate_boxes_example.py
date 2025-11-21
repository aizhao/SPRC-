#!/usr/bin/env python
"""
生成Bounding Box数据的示例脚本

提供三种方式生成boxes:
1. 使用YOLO目标检测
2. 使用显著性检测
3. 使用随机boxes（仅用于测试）
"""

import json
import os
from pathlib import Path
import sys

def generate_random_boxes(image_names, num_boxes_per_image=2):
    """
    生成随机boxes（仅用于测试）
    
    Args:
        image_names: 图像名称列表
        num_boxes_per_image: 每张图像的box数量
    
    Returns:
        boxes字典
    """
    import random
    
    boxes = {}
    for img_name in image_names:
        img_boxes = []
        for _ in range(num_boxes_per_image):
            # 生成随机box [x1, y1, x2, y2]，归一化到[0,1]
            x1 = random.uniform(0, 0.7)
            y1 = random.uniform(0, 0.7)
            x2 = x1 + random.uniform(0.1, 0.3)
            y2 = y1 + random.uniform(0.1, 0.3)
            
            # 确保在[0,1]范围内
            x2 = min(x2, 1.0)
            y2 = min(y2, 1.0)
            
            img_boxes.append([x1, y1, x2, y2])
        
        boxes[img_name] = img_boxes
    
    return boxes


def generate_boxes_with_yolo(image_dir, image_names, conf_threshold=0.5):
    """
    使用YOLO生成boxes
    
    Args:
        image_dir: 图像目录
        image_names: 图像名称列表
        conf_threshold: 置信度阈值
    
    Returns:
        boxes字典
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics未安装，请运行: pip install ultralytics")
        return None
    
    print("加载YOLO模型...")
    model = YOLO('yolov8n.pt')  # 使用nano模型，速度快
    
    boxes = {}
    print(f"处理 {len(image_names)} 张图像...")
    
    for i, img_name in enumerate(image_names):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(image_names)}")
        
        # 构建图像路径
        img_path = Path(image_dir) / img_name
        if not img_path.exists():
            # 尝试添加扩展名
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = Path(image_dir) / f"{img_name}{ext}"
                if test_path.exists():
                    img_path = test_path
                    break
        
        if not img_path.exists():
            print(f"⚠️  图像不存在: {img_path}")
            boxes[img_name] = []
            continue
        
        # 运行检测
        results = model(str(img_path), verbose=False)
        
        # 提取boxes（归一化坐标）
        img_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.conf[0] >= conf_threshold:
                    # xyxyn: 归一化的[x1, y1, x2, y2]
                    coords = box.xyxyn[0].cpu().numpy().tolist()
                    img_boxes.append(coords)
        
        boxes[img_name] = img_boxes
    
    print(f"✅ 完成！共处理 {len(boxes)} 张图像")
    return boxes


def generate_boxes_with_saliency(image_dir, image_names):
    """
    使用显著性检测生成boxes
    
    Args:
        image_dir: 图像目录
        image_names: 图像名称列表
    
    Returns:
        boxes字典
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("❌ opencv未安装，请运行: pip install opencv-python")
        return None
    
    boxes = {}
    print(f"处理 {len(image_names)} 张图像...")
    
    for i, img_name in enumerate(image_names):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(image_names)}")
        
        # 构建图像路径
        img_path = Path(image_dir) / img_name
        if not img_path.exists():
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = Path(image_dir) / f"{img_name}{ext}"
                if test_path.exists():
                    img_path = test_path
                    break
        
        if not img_path.exists():
            boxes[img_name] = []
            continue
        
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            boxes[img_name] = []
            continue
        
        h, w = img.shape[:2]
        
        # 使用静态显著性检测
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(img)
        
        if not success:
            boxes[img_name] = []
            continue
        
        # 二值化
        _, binary_map = cv2.threshold((saliency_map * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 提取最大的几个区域
        img_boxes = []
        if len(contours) > 0:
            # 按面积排序
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # 取前3个最大的区域
            for contour in contours[:3]:
                x, y, bw, bh = cv2.boundingRect(contour)
                
                # 归一化
                x1 = x / w
                y1 = y / h
                x2 = (x + bw) / w
                y2 = (y + bh) / h
                
                # 过滤太小的区域
                if (x2 - x1) * (y2 - y1) > 0.01:  # 面积大于1%
                    img_boxes.append([x1, y1, x2, y2])
        
        boxes[img_name] = img_boxes
    
    print(f"✅ 完成！共处理 {len(boxes)} 张图像")
    return boxes


def main():
    """主函数"""
    print("=" * 60)
    print("Bounding Box生成工具")
    print("=" * 60)
    print()
    
    # 配置
    base_path = Path(__file__).parent
    cirr_dataset_path = base_path / 'cirr_dataset' / 'cirr'
    
    # 检查数据集是否存在
    if not cirr_dataset_path.exists():
        print(f"❌ CIRR数据集不存在: {cirr_dataset_path}")
        print("请确保数据集路径正确")
        return
    
    # 加载图像名称
    print("加载CIRR训练集图像名称...")
    split_file = cirr_dataset_path / 'image_splits' / 'split.rc2.train.json'
    
    if not split_file.exists():
        print(f"❌ 分割文件不存在: {split_file}")
        return
    
    with open(split_file, 'r') as f:
        name_to_relpath = json.load(f)
    
    image_names = list(name_to_relpath.keys())
    print(f"✓ 加载了 {len(image_names)} 张图像")
    
    # 选择生成方式
    print("\n请选择box生成方式:")
    print("1. 随机boxes（仅用于测试）")
    print("2. YOLO目标检测（需要安装ultralytics）")
    print("3. 显著性检测（需要安装opencv）")
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    boxes = None
    output_file = None
    
    if choice == '1':
        print("\n使用随机boxes...")
        boxes = generate_random_boxes(image_names[:100], num_boxes_per_image=2)  # 仅生成前100张
        output_file = base_path / 'cirr_boxes_random.json'
    
    elif choice == '2':
        print("\n使用YOLO检测...")
        image_dir = base_path / 'cirr_dataset'
        boxes = generate_boxes_with_yolo(image_dir, image_names[:100])  # 仅处理前100张作为示例
        output_file = base_path / 'cirr_boxes_yolo.json'
    
    elif choice == '3':
        print("\n使用显著性检测...")
        image_dir = base_path / 'cirr_dataset'
        boxes = generate_boxes_with_saliency(image_dir, image_names[:100])  # 仅处理前100张作为示例
        output_file = base_path / 'cirr_boxes_saliency.json'
    
    else:
        print("❌ 无效的选项")
        return
    
    if boxes is None:
        print("❌ Box生成失败")
        return
    
    # 保存结果
    print(f"\n保存boxes到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(boxes, f, indent=2)
    
    # 统计信息
    total_boxes = sum(len(b) for b in boxes.values())
    avg_boxes = total_boxes / len(boxes) if boxes else 0
    
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"图像数量: {len(boxes)}")
    print(f"总box数量: {total_boxes}")
    print(f"平均每张图像: {avg_boxes:.2f} 个boxes")
    print(f"有boxes的图像: {sum(1 for b in boxes.values() if len(b) > 0)}")
    print(f"无boxes的图像: {sum(1 for b in boxes.values() if len(b) == 0)}")
    
    print("\n✅ 完成！")
    print(f"\n使用方法:")
    print(f"python blip_fine_tune_2.py \\")
    print(f"    --dataset CIRR \\")
    print(f"    --use-region-loss \\")
    print(f"    --box-file {output_file.name} \\")
    print(f"    --loss-region 0.5 \\")
    print(f"    --save-training")


if __name__ == "__main__":
    main()
