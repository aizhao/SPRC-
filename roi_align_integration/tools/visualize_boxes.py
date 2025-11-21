#!/usr/bin/env python
"""
可视化Bounding Boxes的脚本
"""

import json
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_boxes(image_path, boxes, output_path=None):
    """
    在图像上绘制bounding boxes
    
    Args:
        image_path: 图像路径
        boxes: box列表 [[x1,y1,x2,y2], ...]
        output_path: 输出路径（可选）
    """
    # 读取图像
    img = Image.open(image_path)
    width, height = img.size
    
    # 创建图形
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # 绘制每个box
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # 转换为像素坐标
        x1_px = x1 * width
        y1_px = y1 * height
        x2_px = x2 * width
        y2_px = y2 * height
        
        # 计算宽高
        box_width = x2_px - x1_px
        box_height = y2_px - y1_px
        
        # 绘制矩形
        rect = patches.Rectangle(
            (x1_px, y1_px), box_width, box_height,
            linewidth=3, edgecolor=colors[i % len(colors)],
            facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)
        
        # 添加标签
        ax.text(x1_px, y1_px - 5, f'Box {i+1}',
                color=colors[i % len(colors)],
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.axis('off')
    plt.title(f'Bounding Boxes Visualization\nImage: {Path(image_path).name}', 
              fontsize=14, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"✓ 保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("Bounding Box可视化工具")
    print("=" * 60)
    
    # 配置
    base_path = Path(__file__).parent
    
    # 加载box文件
    box_files = list(base_path.glob('../data/cirr_boxes_train.json'))
    print(box_files)
    if not box_files:
        print("❌ 未找到box文件")
        print("请先运行: python generate_boxes_example.py")
        return
    
    print(f"\n找到 {len(box_files)} 个box文件:")
    for i, f in enumerate(box_files, 1):
        print(f"  {i}. {f.name}")
    
    # 选择文件
    choice = input(f"\n请选择文件 (1-{len(box_files)}): ").strip()
    try:
        box_file = box_files[int(choice) - 1]
    except:
        print("❌ 无效选择")
        return
    
    print(f"\n加载: {box_file}")
    with open(box_file, 'r') as f:
        boxes_data = json.load(f)
    
    print(f"✓ 加载了 {len(boxes_data)} 张图像的boxes")
    
    # 选择要可视化的图像
    image_names = [name for name, boxes in boxes_data.items() if len(boxes) > 0]
    
    if not image_names:
        print("❌ 没有图像有boxes")
        return
    
    print(f"\n有boxes的图像数量: {len(image_names)}")
    print("显示前5张:")
    for i, name in enumerate(image_names[:5], 1):
        num_boxes = len(boxes_data[name])
        print(f"  {i}. {name} ({num_boxes} boxes)")
    
    # 可视化前几张
    num_to_show = min(5, len(image_names))
    output_dir = base_path / 'box_visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n开始可视化前 {num_to_show} 张图像...")
    
    # CIRR数据集路径 - 从项目根目录查找
    project_root = base_path.parent.parent
    cirr_path = project_root / 'cirr_dataset'
    print(cirr_path)
    if not cirr_path.exists():
        print(f"❌ CIRR数据集路径不存在: {cirr_path}")
        print("请设置正确的CIRR数据集路径")
        return
    
    for i, img_name in enumerate(image_names[:num_to_show], 1):
        # 查找图像文件 - CIRR图像在dev/train等子目录中
        img_path = None
        for split in ['train']:
            for num in range(0, 100):
                for ext in ['.png']:
                    test_path = cirr_path / split / str(num) / f"{img_name}{ext}"
                    if test_path.exists():
                        img_path = test_path
                        break
                if img_path:
                    break
            if img_path:
                break
        
        if img_path is None or not img_path.exists():
            print(f"  ⚠️  图像不存在: {img_name}")
            continue
        
        boxes = boxes_data[img_name]
        output_path = output_dir / f"{img_name}_boxes.png"
        
        try:
            visualize_boxes(img_path, boxes, output_path)
            print(f"  {i}. ✓ {img_name} ({len(boxes)} boxes)")
        except Exception as e:
            print(f"  {i}. ❌ {img_name}: {str(e)}")
    
    print(f"\n✅ 完成！可视化结果保存在: {output_dir}")
    print(f"\n查看结果:")
    print(f"  ls {output_dir}")


if __name__ == "__main__":
    try:
        import matplotlib
        main()
    except ImportError:
        print("❌ 需要安装matplotlib")
        print("运行: pip install matplotlib")
