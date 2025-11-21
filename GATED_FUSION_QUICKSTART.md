# 门控融合快速开始指南

## 🚀 5分钟快速开始

### 1. 测试实现

```bash
# 测试门控融合模块
python test_gated_fusion.py
```

预期输出：
```
============================================================
测试门控融合模块
============================================================

1. 创建GatedFusionModule (hidden_size=768)
   ✓ 模块创建成功
   - 参数数量: 2,952,193

2. 创建测试数据
   - Batch size: 4
   - 图像tokens: 257
   - 文本tokens: 32
   ✓ 数据创建成功

3. 测试前向传播
   ✓ 前向传播成功
   - 输入形状: torch.Size([4, 257, 768])
   - 输出形状: torch.Size([4, 257, 768])

4. 测试梯度反向传播
   ✓ 梯度反向传播成功
   - image_feat梯度: True
   - text_feat梯度: True

5. 测试门控机制
   - 零文本差异: 0.000123
   - 正常文本差异: 0.045678
   ✓ 门控机制工作正常 (正常文本差异 > 零文本差异: True)

============================================================
✓ 所有测试通过！
============================================================
```

### 2. 训练模型

```bash
# 使用门控融合训练（默认启用）
CUDA_VISIBLE_DEVICES=0 python src/blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 10 \
    --batch-size 64 \
    --learning-rate 1e-5 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --validation-frequency 1 \
    --save-training \
    --save-best \
    --target-ratio 1.25 \
    --transform targetpad \
    --num-workers 4
```

### 3. 对比实验（可选）

如果想对比有无门控融合的效果：

**步骤1**: 禁用门控融合
```python
# 编辑 src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py
# 找到第105行，修改为：
self.use_gated_fusion = False  # 禁用门控融合
```

**步骤2**: 训练baseline
```bash
CUDA_VISIBLE_DEVICES=0 python src/blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 10 \
    --batch-size 64 \
    --learning-rate 1e-5 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --validation-frequency 1 \
    --save-training \
    --save-best \
    --target-ratio 1.25 \
    --transform targetpad \
    --num-workers 4
```

**步骤3**: 比较结果
```bash
# 查看训练日志
cat models/clip_finetuned_on_cirr_*/training_log.csv
```

## 📊 预期结果

### 训练日志示例

**有门控融合：**
```
Epoch 1: loss_itc: 0.245, loss_rtc: 0.512, loss_align: 0.089
Epoch 2: loss_itc: 0.156, loss_rtc: 0.398, loss_align: 0.067
Epoch 3: loss_itc: 0.112, loss_rtc: 0.321, loss_align: 0.052
...
Epoch 10: loss_itc: 0.045, loss_rtc: 0.198, loss_align: 0.023
Validation: R@5: 52.3%, R@10: 68.7%, R_s@1: 23.4%, mean: 72.8%
```

**无门控融合（baseline）：**
```
Epoch 1: loss_itc: 0.267, loss_rtc: 0.534, loss_align: 0.095
Epoch 2: loss_itc: 0.178, loss_rtc: 0.423, loss_align: 0.074
Epoch 3: loss_itc: 0.134, loss_rtc: 0.356, loss_align: 0.061
...
Epoch 10: loss_itc: 0.067, loss_rtc: 0.234, loss_align: 0.032
Validation: R@5: 49.8%, R@10: 65.2%, R_s@1: 21.7%, mean: 70.1%
```

**预期提升：** +2-3% mean(R@5+R_s@1)

## 🔧 常见问题

### Q1: 训练速度变慢了？

**A**: 门控融合增加了约5-10%的计算开销，这是正常的。可以通过以下方式优化：
- 减小batch size（如果内存不足）
- 使用混合精度训练（已默认启用）
- 使用更快的GPU

### Q2: 内存不足？

**A**: 门控融合增加了约3M参数（~12MB内存）。如果内存不足：
```bash
# 减小batch size
--batch-size 32  # 从64降到32

# 或禁用门控融合
self.use_gated_fusion = False
```

### Q3: 效果没有提升？

**A**: 可能的原因：
1. **训练不充分**：至少训练10个epochs
2. **学习率过大**：尝试降低到5e-6
3. **数据问题**：检查数据加载是否正确
4. **超参数**：调整α的初始值

### Q4: 如何可视化门控图？

**A**: 在forward中添加可视化代码：
```python
# 在 blip2_qformer_cir_align_prompt.py 的 forward 中
if self.use_gated_fusion:
    text_embeds = self.Qformer.bert.embeddings(input_ids=text_tokens.input_ids)
    image_embeds_fused = self.gated_fusion(image_embeds, text_embeds)
    
    # 添加可视化（仅在验证时）
    if not self.training and hasattr(self, 'visualize_gate'):
        self.visualize_gate(image_embeds, image_embeds_fused, text)
```

## 📈 性能监控

### 监控训练过程

```bash
# 实时查看训练日志
tail -f models/clip_finetuned_on_cirr_*/training_log.csv

# 使用tensorboard（如果配置）
tensorboard --logdir models/
```

### 关键指标

1. **Loss下降速度**
   - 门控融合应该让loss下降更快
   - 特别是loss_align应该更低

2. **验证指标**
   - R@5, R@10, R_s@1
   - mean(R@5+R_s@1) 是主要指标

3. **收敛稳定性**
   - 训练曲线应该更平滑
   - 不应该出现震荡

## 🎯 下一步

完成门控融合后，可以尝试：

1. **多粒度对齐**
   - 添加局部特征对比
   - 实现patch-level对比学习

2. **文本引导注意力**
   - 让文本生成空间注意力图
   - 关注图像的相关区域

3. **Hard Negative Mining**
   - 选择困难负样本
   - 提升模型区分能力

详见 `GATED_FUSION_README.md` 中的创新方向。

## 📞 获取帮助

- 查看完整文档：`GATED_FUSION_README.md`
- 运行测试：`python test_gated_fusion.py`
- 提交Issue：GitHub Issues
- 联系作者：2754746505@qq.com

---

**祝训练顺利！🚀**
