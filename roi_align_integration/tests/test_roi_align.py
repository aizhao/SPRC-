#!/usr/bin/env python
"""
RoI Align功能测试脚本

测试新添加的区域特征提取功能是否正常工作
"""

import sys
import os

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

import torch
from lavis.models import load_model_and_preprocess

def load_test_model(device="cuda"):
    """加载测试模型并处理类型转换"""
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_cir_align_prompt", 
        model_type="pretrain", 
        is_eval=False, 
        device=device
    )
    
    # 将模型转换为float32以避免类型不匹配
    # 无论CPU还是CUDA都转换为float32
    model = model.float()
    
    return model, vis_processors, txt_processors

def test_basic_forward():
    """测试基本的前向传播（不使用区域损失）"""
    print("=" * 60)
    print("测试1: 基本前向传播（不使用区域损失）")
    print("=" * 60)
    
    # 加载模型
    print("加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_test_model(device)
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    target = torch.randn(batch_size, 3, 224, 224).to(device)
    text = ["a red car", "a blue shirt"]
    
    # 前向传播
    print("执行前向传播...")
    model.train()
    samples = {
        "image": image,
        "target": target,
        "text_input": text,
        "region_boxes": None,
        "target_region_boxes": None
    }
    
    losses = model(samples)
    
    print("\n损失值:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✅ 测试1通过！")
    return True


def test_region_loss():
    """测试区域损失功能"""
    print("\n" + "=" * 60)
    print("测试2: 区域损失功能")
    print("=" * 60)
    
    # 加载模型
    print("加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_test_model(device)
    
    # 启用区域损失
    model.use_region_loss = True
    print("✓ 区域损失已启用")
    
    # 创建测试数据
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    target = torch.randn(batch_size, 3, 224, 224).to(device)
    text = ["a red car", "a blue shirt"]
    
    # 创建测试boxes（归一化坐标 [0, 1]）
    ref_boxes = [
        [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]],  # 第一张图的2个boxes
        [[0.2, 0.2, 0.8, 0.8]]  # 第二张图的1个box
    ]
    tgt_boxes = [
        [[0.15, 0.15, 0.55, 0.55], [0.65, 0.65, 0.95, 0.95]],  # 对应的目标boxes
        [[0.25, 0.25, 0.85, 0.85]]
    ]
    
    print(f"\n参考图像boxes数量: {[len(b) for b in ref_boxes]}")
    print(f"目标图像boxes数量: {[len(b) for b in tgt_boxes]}")
    
    # 前向传播
    print("\n执行前向传播...")
    model.train()
    samples = {
        "image": image,
        "target": target,
        "text_input": text,
        "region_boxes": ref_boxes,
        "target_region_boxes": tgt_boxes
    }
    
    losses = model(samples)
    
    print("\n损失值:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # 验证区域损失是否存在
    assert 'loss_region' in losses, "❌ 区域损失未计算！"
    assert losses['loss_region'].item() > 0, "❌ 区域损失为0！"
    
    print("\n✅ 测试2通过！区域损失正常工作")
    return True


def test_extract_region_features():
    """测试区域特征提取功能"""
    print("\n" + "=" * 60)
    print("测试3: 区域特征提取")
    print("=" * 60)
    
    # 加载模型
    print("加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_test_model(device)
    
    # 创建模拟的图像特征 (B, N, D)
    # 假设是14x14的feature map，hidden_dim=768
    batch_size = 2
    feature_map_size = 14
    hidden_dim = 768
    num_patches = feature_map_size * feature_map_size
    
    image_embeds = torch.randn(batch_size, num_patches, hidden_dim).to(device)
    
    # 创建测试boxes
    boxes = [
        [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]],  # 2个boxes
        [[0.2, 0.2, 0.8, 0.8]]  # 1个box
    ]
    
    print(f"\n输入特征形状: {image_embeds.shape}")
    print(f"Boxes: {boxes}")
    
    # 提取区域特征
    print("\n提取区域特征...")
    region_features = model.extract_region_features(image_embeds, boxes)
    
    print(f"\n提取的区域特征:")
    for i, feats in enumerate(region_features):
        print(f"  图像 {i}: {feats.shape} (num_boxes={feats.shape[0]}, dim={feats.shape[1]})")
    
    # 验证
    assert len(region_features) == batch_size, "❌ 区域特征数量不对！"
    assert region_features[0].shape[0] == 2, "❌ 第一张图应该有2个区域！"
    assert region_features[1].shape[0] == 1, "❌ 第二张图应该有1个区域！"
    assert region_features[0].shape[1] == hidden_dim, "❌ 特征维度不对！"
    
    print("\n✅ 测试3通过！区域特征提取正常")
    return True


def test_empty_boxes():
    """测试空boxes的情况"""
    print("\n" + "=" * 60)
    print("测试4: 空boxes处理")
    print("=" * 60)
    
    # 加载模型
    print("加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_test_model(device)
    model.use_region_loss = True
    
    # 创建测试数据
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    target = torch.randn(batch_size, 3, 224, 224).to(device)
    text = ["a red car", "a blue shirt"]
    
    # 一个有boxes，一个没有
    ref_boxes = [
        [[0.1, 0.1, 0.5, 0.5]],  # 有1个box
        []  # 空boxes
    ]
    tgt_boxes = [
        [[0.15, 0.15, 0.55, 0.55]],
        []
    ]
    
    print(f"\n参考图像boxes: {ref_boxes}")
    
    # 前向传播
    print("\n执行前向传播...")
    model.train()
    samples = {
        "image": image,
        "target": target,
        "text_input": text,
        "region_boxes": ref_boxes,
        "target_region_boxes": tgt_boxes
    }
    
    losses = model(samples)
    
    print("\n损失值:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✅ 测试4通过！空boxes处理正常")
    return True


def test_different_box_numbers():
    """测试不同数量的boxes"""
    print("\n" + "=" * 60)
    print("测试5: 不同数量的boxes")
    print("=" * 60)
    
    # 加载模型
    print("加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_test_model(device)
    model.use_region_loss = True
    
    # 创建测试数据
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    target = torch.randn(batch_size, 3, 224, 224).to(device)
    text = ["a red car", "a blue shirt"]
    
    # 参考图像和目标图像的boxes数量不同
    ref_boxes = [
        [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]],  # 2个boxes
        [[0.2, 0.2, 0.8, 0.8]]  # 1个box
    ]
    tgt_boxes = [
        [[0.15, 0.15, 0.55, 0.55]],  # 只有1个box（数量不同）
        [[0.25, 0.25, 0.85, 0.85], [0.5, 0.5, 0.9, 0.9]]  # 2个boxes（数量不同）
    ]
    
    print(f"\n参考图像boxes数量: {[len(b) for b in ref_boxes]}")
    print(f"目标图像boxes数量: {[len(b) for b in tgt_boxes]}")
    
    # 前向传播
    print("\n执行前向传播...")
    model.train()
    samples = {
        "image": image,
        "target": target,
        "text_input": text,
        "region_boxes": ref_boxes,
        "target_region_boxes": tgt_boxes
    }
    
    losses = model(samples)
    
    print("\n损失值:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✅ 测试5通过！不同数量boxes处理正常")
    return True


def main():
    """运行所有测试"""
    print("\n" + "🔬" * 30)
    print("RoI Align功能测试套件")
    print("🔬" * 30 + "\n")
    
    tests = [
        ("基本前向传播", test_basic_forward),
        ("区域损失功能", test_region_loss),
        ("区域特征提取", test_extract_region_features),
        ("空boxes处理", test_empty_boxes),
        ("不同数量boxes", test_different_box_numbers),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ 测试失败: {test_name}")
            print(f"错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"✅ 通过: {passed}/{len(tests)}")
    print(f"❌ 失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！RoI Align功能正常工作！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    exit(main())
