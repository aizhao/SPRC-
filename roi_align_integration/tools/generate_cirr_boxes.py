#!/usr/bin/env python
"""
CIRR数据集Bounding Box生成模块

功能:
1. 自动为CIRR数据集中的所有图像生成bounding boxes
2. 支持多种检测方法（YOLO、显著性检测、组合方法）
3. 智能过滤和优化boxes
4. 生成训练和验证集的boxes
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import sys
from tqdm import tqdm
import argparse


class CIRRBoxGenerator:
    """CIRR数据集Box生成器"""
    
    def __init__(self, cirr_root: str, method: str = 'yolo'):
        """
        初始化生成器
        
        Args:
            cirr_root: CIRR数据集根目录
            method: 检测方法 ('yolo', 'saliency', 'hybrid')
        """
        self.cirr_root = Path(cirr_root)
        self.method = method
        self.boxes_cache = {}
        
        # 检查数据集是否存在
        if not self.cirr_root.exists():
            raise ValueError(f"CIRR数据集不存在: {self.cirr_root}")
        
        print(f"✓ CIRR数据集路径: {self.cirr_root}")
        print(f"✓ 检测方法: {method}")
    
    def load_image_list(self, split: str = 'train') -> Dict[str, str]:
        """
        加载指定split的图像列表
        
        Args:
            split: 'train', 'val', 或 'test1'
        
        Returns:
            {image_name: relative_path} 字典
        """
        split_file = self.cirr_root / 'cirr' / 'image_splits' / f'split.rc2.{split}.json'
        
        if not split_file.exists():
            raise ValueError(f"分割文件不存在: {split_file}")
        
        with open(split_file, 'r') as f:
            name_to_relpath = json.load(f)
        
        print(f"✓ 加载 {split} split: {len(name_to_relpath)} 张图像")
        return name_to_relpath
    
    def get_image_path(self, image_name: str, name_to_relpath: Dict[str, str]) -> Path:
        """获取图像的完整路径"""
        rel_path = name_to_relpath.get(image_name)
        if rel_path is None:
            return None
        return self.cirr_root / rel_path
    
    def detect_with_yolo(self, image_path: Path, conf_threshold: float = 0.3) -> List[List[float]]:
        """
        使用YOLO检测物体
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
        
        Returns:
            boxes列表 [[x1, y1, x2, y2], ...]
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装ultralytics: pip install ultralytics")
        
        if not hasattr(self, 'yolo_model'):
            print("加载YOLO模型...")
            self.yolo_model = YOLO('yolov8n.pt')  # 使用nano模型
        
        # 运行检测
        results = self.yolo_model(str(image_path), verbose=False)
        
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.conf[0] >= conf_threshold:
                    # xyxyn: 归一化的[x1, y1, x2, y2]
                    coords = box.xyxyn[0].cpu().numpy().tolist()
                    boxes.append(coords)
        
        return boxes
    
    def detect_with_saliency(self, image_path: Path, num_regions: int = 3) -> List[List[float]]:
        """
        使用显著性检测找关键区域
        
        Args:
            image_path: 图像路径
            num_regions: 最多返回的区域数
        
        Returns:
            boxes列表
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise ImportError("请安装opencv: pip install opencv-python opencv-contrib-python")
        
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        h, w = img.shape[:2]
        
        # 使用静态显著性检测
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(img)
        
        if not success:
            return []
        
        # 二值化
        _, binary_map = cv2.threshold(
            (saliency_map * 255).astype(np.uint8), 
            127, 255, cv2.THRESH_BINARY
        )
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return []
        
        # 按面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        boxes = []
        for contour in contours[:num_regions]:
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # 归一化
            x1 = x / w
            y1 = y / h
            x2 = (x + bw) / w
            y2 = (y + bh) / h
            
            # 过滤太小的区域（面积小于1%）
            if (x2 - x1) * (y2 - y1) > 0.01:
                boxes.append([x1, y1, x2, y2])
        
        return boxes
    
    def detect_hybrid(self, image_path: Path) -> List[List[float]]:
        """
        混合方法：结合YOLO和显著性检测
        
        优先使用YOLO，如果检测不到物体则使用显著性检测
        """
        # 先尝试YOLO
        boxes = self.detect_with_yolo(image_path, conf_threshold=0.3)
        
        # 如果YOLO没检测到，使用显著性检测
        if len(boxes) == 0:
            boxes = self.detect_with_saliency(image_path, num_regions=2)
        
        return boxes
    
    def filter_boxes(self, boxes: List[List[float]], 
                    min_area: float = 0.01, 
                    max_boxes: int = 5) -> List[List[float]]:
        """
        过滤和优化boxes
        
        Args:
            boxes: 原始boxes
            min_area: 最小面积阈值
            max_boxes: 最多保留的boxes数量
        
        Returns:
            过滤后的boxes
        """
        if not boxes:
            return []
        
        # 过滤太小的boxes
        filtered = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                filtered.append(box)
        
        # 按面积排序，保留最大的几个
        filtered.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        
        return filtered[:max_boxes]
    
    def generate_boxes_for_split(self, split: str = 'train', 
                                 max_images: int = None,
                                 save_interval: int = 100) -> Dict[str, List]:
        """
        为指定split生成boxes
        
        Args:
            split: 数据集分割
            max_images: 最多处理的图像数（None表示全部）
            save_interval: 每处理多少张图像保存一次
        
        Returns:
            boxes字典
        """
        print(f"\n{'='*60}")
        print(f"为 {split} split 生成boxes")
        print(f"{'='*60}")
        
        # 加载图像列表
        name_to_relpath = self.load_image_list(split)
        image_names = list(name_to_relpath.keys())
        
        if max_images:
            image_names = image_names[:max_images]
            print(f"限制处理数量: {max_images}")
        
        boxes_dict = {}
        failed_count = 0
        
        # 选择检测方法
        if self.method == 'yolo':
            detect_func = self.detect_with_yolo
        elif self.method == 'saliency':
            detect_func = self.detect_with_saliency
        elif self.method == 'hybrid':
            detect_func = self.detect_hybrid
        else:
            raise ValueError(f"未知的检测方法: {self.method}")
        
        # 处理每张图像
        print(f"\n开始处理 {len(image_names)} 张图像...")
        
        for i, img_name in enumerate(tqdm(image_names, desc="生成boxes")):
            try:
                # 获取图像路径
                img_path = self.get_image_path(img_name, name_to_relpath)
                
                if img_path is None or not img_path.exists():
                    boxes_dict[img_name] = []
                    failed_count += 1
                    continue
                
                # 检测boxes
                boxes = detect_func(img_path)
                
                # 过滤和优化
                boxes = self.filter_boxes(boxes, min_area=0.01, max_boxes=5)
                
                boxes_dict[img_name] = boxes
                
                # 定期保存
                if save_interval and (i + 1) % save_interval == 0:
                    self._save_checkpoint(boxes_dict, split, i + 1)
            
            except Exception as e:
                print(f"\n⚠️  处理失败 {img_name}: {str(e)}")
                boxes_dict[img_name] = []
                failed_count += 1
        
        # 统计信息
        total = len(boxes_dict)
        with_boxes = sum(1 for b in boxes_dict.values() if len(b) > 0)
        total_boxes = sum(len(b) for b in boxes_dict.values())
        
        print(f"\n{'='*60}")
        print(f"统计信息 - {split}")
        print(f"{'='*60}")
        print(f"总图像数: {total}")
        print(f"有boxes的图像: {with_boxes} ({with_boxes/total*100:.1f}%)")
        print(f"无boxes的图像: {total - with_boxes}")
        print(f"总box数: {total_boxes}")
        print(f"平均每张图像: {total_boxes/total:.2f} boxes")
        print(f"处理失败: {failed_count}")
        
        return boxes_dict
    
    def _save_checkpoint(self, boxes_dict: Dict, split: str, count: int):
        """保存中间结果"""
        checkpoint_file = Path(f'cirr_boxes_{split}_checkpoint_{count}.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(boxes_dict, f, indent=2)
        print(f"\n✓ 保存检查点: {checkpoint_file}")
    
    def save_boxes(self, boxes_dict: Dict, output_file: str):
        """保存boxes到文件"""
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(boxes_dict, f, indent=2)
        
        print(f"\n✅ Boxes已保存到: {output_path}")
        print(f"文件大小: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='为CIRR数据集生成bounding boxes')
    parser.add_argument('--cirr-root', type=str, default='./cirr_dataset',
                       help='CIRR数据集根目录')
    parser.add_argument('--method', type=str, default='yolo',
                       choices=['yolo', 'saliency', 'hybrid'],
                       help='检测方法')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test1'],
                       help='数据集分割')
    parser.add_argument('--max-images', type=int, default=None,
                       help='最多处理的图像数（用于测试）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件名')
    parser.add_argument('--all-splits', action='store_true',
                       help='处理所有splits（train和val）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CIRR数据集Bounding Box生成工具")
    print("=" * 60)
    print()
    
    try:
        # 创建生成器
        generator = CIRRBoxGenerator(args.cirr_root, args.method)
        
        if args.all_splits:
            # 处理所有splits
            for split in ['train', 'val']:
                boxes = generator.generate_boxes_for_split(
                    split=split,
                    max_images=args.max_images,
                    save_interval=100
                )
                
                output_file = f'cirr_boxes_{args.method}_{split}.json'
                generator.save_boxes(boxes, output_file)
        else:
            # 处理单个split
            boxes = generator.generate_boxes_for_split(
                split=args.split,
                max_images=args.max_images,
                save_interval=100
            )
            
            # 保存结果
            if args.output:
                output_file = args.output
            else:
                output_file = f'cirr_boxes_{args.method}_{args.split}.json'
            
            generator.save_boxes(boxes, output_file)
        
        print("\n" + "=" * 60)
        print("✅ 完成！")
        print("=" * 60)
        print("\n使用方法:")
        print(f"python blip_fine_tune_2.py \\")
        print(f"    --dataset CIRR \\")
        print(f"    --use-region-loss \\")
        print(f"    --box-file {output_file} \\")
        print(f"    --save-training")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
