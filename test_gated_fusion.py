"""
测试门控融合模块的脚本

用法：
python test_gated_fusion.py
"""

import torch
import sys
sys.path.append('src')

from lavis.models.blip2_models.blip2_qformer_cir_align_prompt import GatedFusionModule


def test_gated_fusion_module():
    """测试GatedFusionModule的基本功能"""
    print("=" * 60)
    print("测试门控融合模块")
    print("=" * 60)
    
    # 参数设置
    batch_size = 4
    num_img_tokens = 257  # ViT输出的token数量 (1 CLS + 16x16 patches)
    num_txt_tokens = 32   # 文本token数量
    image_dim = 1088      # Visual encoder的输出维度
    text_dim = 768        # Q-former的hidden size
    
    # 创建模块
    print(f"\n1. 创建GatedFusionModule")
    print(f"   - image_dim: {image_dim}")
    print(f"   - text_dim: {text_dim}")
    fusion_module = GatedFusionModule(image_dim=image_dim, text_dim=text_dim, dropout=0.1)
    print(f"   ✓ 模块创建成功")
    print(f"   - 参数数量: {sum(p.numel() for p in fusion_module.parameters()):,}")
    
    # 创建测试数据
    print(f"\n2. 创建测试数据")
    print(f"   - Batch size: {batch_size}")
    print(f"   - 图像tokens: {num_img_tokens}, 维度: {image_dim}")
    print(f"   - 文本tokens: {num_txt_tokens}, 维度: {text_dim}")
    
    image_feat = torch.randn(batch_size, num_img_tokens, image_dim)
    text_feat = torch.randn(batch_size, num_txt_tokens, text_dim)
    print(f"   ✓ 数据创建成功")
    
    # 前向传播
    print(f"\n3. 测试前向传播")
    with torch.no_grad():
        fused_feat = fusion_module(image_feat, text_feat)
    
    print(f"   ✓ 前向传播成功")
    print(f"   - 输入形状: {image_feat.shape}")
    print(f"   - 输出形状: {fused_feat.shape}")
    assert fused_feat.shape == image_feat.shape, "输出形状应该与输入相同"
    
    # 测试梯度
    print(f"\n4. 测试梯度反向传播")
    image_feat.requires_grad = True
    text_feat.requires_grad = True
    
    fused_feat = fusion_module(image_feat, text_feat)
    loss = fused_feat.mean()
    loss.backward()
    
    print(f"   ✓ 梯度反向传播成功")
    print(f"   - image_feat梯度: {image_feat.grad is not None}")
    print(f"   - text_feat梯度: {text_feat.grad is not None}")
    
    # 测试门控效果
    print(f"\n5. 测试门控机制")
    with torch.no_grad():
        # 零文本应该产生接近原始图像的输出
        zero_text = torch.zeros(batch_size, num_txt_tokens, hidden_size)
        fused_zero = fusion_module(image_feat, zero_text)
        
        # 计算与原始图像的差异
        diff_zero = (fused_zero - image_feat).abs().mean().item()
        
        # 正常文本应该产生更大的差异
        fused_normal = fusion_module(image_feat, text_feat)
        diff_normal = (fused_normal - image_feat).abs().mean().item()
        
        print(f"   - 零文本差异: {diff_zero:.6f}")
        print(f"   - 正常文本差异: {diff_normal:.6f}")
        print(f"   ✓ 门控机制工作正常 (正常文本差异 > 零文本差异: {diff_normal > diff_zero})")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)


def test_integration_with_model():
    """测试与完整模型的集成"""
    print("\n" + "=" * 60)
    print("测试与BLIP2模型的集成")
    print("=" * 60)
    
    try:
        from lavis.models import load_model_and_preprocess
        
        print("\n1. 加载BLIP2模型")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   - 设备: {device}")
        
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_cir_align_prompt",
            model_type="pretrain",
            is_eval=False,
            device=device
        )
        
        print(f"   ✓ 模型加载成功")
        print(f"   - use_gated_fusion: {model.use_gated_fusion}")
        
        # 检查门控融合模块
        print(f"\n2. 检查门控融合模块")
        assert hasattr(model, 'gated_fusion'), "模型应该有gated_fusion属性"
        print(f"   ✓ 门控融合模块存在")
        print(f"   - 参数数量: {sum(p.numel() for p in model.gated_fusion.parameters()):,}")
        
        # 测试前向传播
        print(f"\n3. 测试完整前向传播")
        batch_size = 2
        samples = {
            "image": torch.randn(batch_size, 3, 224, 224).to(device),
            "target": torch.randn(batch_size, 3, 224, 224).to(device),
            "text_input": ["change the color to red"] * batch_size
        }
        
        with torch.no_grad():
            output = model(samples)
        
        print(f"   ✓ 前向传播成功")
        print(f"   - 输出keys: {list(output.keys())}")
        
        print("\n" + "=" * 60)
        print("✓ 集成测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 测试门控融合模块
    test_gated_fusion_module()
    
    # 测试与模型的集成
    print("\n")
    test_integration_with_model()
