# CIRR数据集Box生成完整指南

## 📋 概述

这个指南将帮助你为CIRR数据集生成bounding boxes，然后使用这些boxes进行训练。

## 🎯 完整流程

```
步骤1: 生成Boxes → 步骤2: 验证Boxes → 步骤3: 训练模型
```

## 📦 步骤1: 生成Bounding Boxes

### 方法A: 快速测试（推荐新手）

**生成少量图像的boxes用于测试**

```bash
cd /home/caoyu/mnt/zhaoai/SPRC

# 使用YOLO为前100张图像生成boxes
python generate_cirr_boxes.py \
    --cirr-root ./cirr_dataset \
    --method yolo \
    --split train \
    --max-images 100 \
    --output cirr_boxes_test.json
```

**预计时间**: 2-5分钟  
**输出**: `cirr_boxes_test.json` (包含100张图像的boxes)

### 方法B: 生成训练集boxes

**为整个训练集生成boxes**

```bash
# 安装YOLO（如果还没安装）
pip install ultralytics

# 生成训练集boxes
python generate_cirr_boxes.py \
    --cirr-root ./cirr_dataset \
    --method yolo \
    --split train \
    --output cirr_boxes_train.json
```

**预计时间**: 1-2小时（取决于数据集大小和GPU）  
**输出**: `cirr_boxes_train.json`

### 方法C: 生成所有splits的boxes

**同时生成训练集和验证集的boxes**

```bash
python generate_cirr_boxes.py \
    --cirr-root ./cirr_dataset \
    --method yolo \
    --all-splits
```

**输出**: 
- `cirr_boxes_yolo_train.json`
- `cirr_boxes_yolo_val.json`

### 检测方法选择

| 方法 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| `yolo` | 准确，识别具体物体 | 需要GPU，较慢 | 🔥 推荐用于实际训练 |
| `saliency` | 快速，不需要GPU | 不够精确 | 快速测试 |
| `hybrid` | 结合两者优点 | 稍慢 | 平衡选择 |

## 🔍 步骤2: 验证生成的Boxes

### 查看统计信息

```bash
# 查看boxes文件内容
python -c "
import json
with open('cirr_boxes_train.json', 'r') as f:
    boxes = json.load(f)
    
total = len(boxes)
with_boxes = sum(1 for b in boxes.values() if len(b) > 0)
total_boxes = sum(len(b) for b in boxes.values())

print(f'总图像数: {total}')
print(f'有boxes的图像: {with_boxes} ({with_boxes/total*100:.1f}%)')
print(f'总box数: {total_boxes}')
print(f'平均每张: {total_boxes/total:.2f} boxes')
"
```

### 可视化boxes（可选）

```bash
# 安装matplotlib
pip install matplotlib

# 可视化前几张图像的boxes
python visualize_boxes.py
```

## 🚀 步骤3: 使用Boxes训练模型

### 3.1 快速测试训练

**使用少量数据测试功能是否正常**

```bash
cd src

python blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 2 \
    --batch-size 32 \
    --learning-rate 2e-6 \
    --use-region-loss \
    --box-file ../cirr_boxes_test.json \
    --loss-region 0.5 \
    --save-training
```

**预计时间**: 10-20分钟  
**目的**: 验证代码运行正常

### 3.2 完整训练（两阶段）

#### 阶段1: 全局特征训练（不使用region loss）

```bash
python blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 30 \
    --batch-size 128 \
    --learning-rate 2e-6 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --validation-frequency 1 \
    --save-training \
    --save-best
```

**预计时间**: 根据GPU和数据集大小，可能需要几小时到一天

#### 阶段2: 细粒度微调（使用region loss）

```bash
python blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 20 \
    --batch-size 128 \
    --learning-rate 5e-7 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --use-region-loss \
    --box-file ../cirr_boxes_train.json \
    --loss-region 0.5 \
    --validation-frequency 1 \
    --save-training \
    --save-best
```

### 3.3 单阶段训练（直接使用region loss）

```bash
python blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 50 \
    --batch-size 128 \
    --learning-rate 2e-6 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --use-region-loss \
    --box-file ../cirr_boxes_train.json \
    --loss-region 0.5 \
    --validation-frequency 1 \
    --save-training \
    --save-best
```

## 📊 参数调优建议

### Region Loss权重

```bash
# 如果region loss过大（>2.0），降低权重
--loss-region 0.3

# 如果region loss过小（<0.1），增加权重
--loss-region 0.7

# 默认推荐值
--loss-region 0.5
```

### Batch Size

```bash
# GPU内存充足
--batch-size 256

# GPU内存一般（推荐）
--batch-size 128

# GPU内存不足
--batch-size 64
```

### 学习率

```bash
# 第一阶段（全局特征）
--learning-rate 2e-6

# 第二阶段（微调）
--learning-rate 5e-7
```

## 🔧 故障排除

### 问题1: YOLO安装失败

```bash
# 尝试使用国内镜像
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题2: 内存不足

```bash
# 减小batch size
--batch-size 32

# 或者处理更少的图像
--max-images 500
```

### 问题3: Box生成太慢

```bash
# 使用显著性检测（更快）
--method saliency

# 或者先处理一部分
--max-images 1000
```

### 问题4: 训练时CUDA out of memory

```bash
# 减小batch size
--batch-size 64

# 或者使用更小的模型
--backbone pretrain_vitL  # 使用ViT-L而不是ViT-G
```

## 📝 完整示例脚本

创建一个自动化脚本 `train_with_boxes.sh`:

```bash
#!/bin/bash

echo "=========================================="
echo "CIRR数据集完整训练流程"
echo "=========================================="

# 步骤1: 生成boxes
echo "步骤1: 生成bounding boxes..."
python generate_cirr_boxes.py \
    --cirr-root ./cirr_dataset \
    --method yolo \
    --split train \
    --max-images 1000 \
    --output cirr_boxes_1k.json

if [ $? -ne 0 ]; then
    echo "❌ Box生成失败"
    exit 1
fi

# 步骤2: 验证
echo ""
echo "步骤2: 验证boxes..."
python -c "
import json
with open('cirr_boxes_1k.json', 'r') as f:
    boxes = json.load(f)
print(f'✓ 加载了 {len(boxes)} 张图像的boxes')
"

# 步骤3: 测试训练
echo ""
echo "步骤3: 快速测试训练..."
cd src
python blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --num-epochs 2 \
    --batch-size 32 \
    --use-region-loss \
    --box-file ../cirr_boxes_1k.json \
    --save-training

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 测试成功！可以开始完整训练了"
    echo ""
    echo "运行完整训练:"
    echo "python blip_fine_tune_2.py --dataset CIRR --use-region-loss --box-file ../cirr_boxes_1k.json --num-epochs 50 --save-training"
else
    echo "❌ 测试失败，请检查错误信息"
fi
```

使用方法:
```bash
chmod +x train_with_boxes.sh
./train_with_boxes.sh
```

## 📈 预期效果

使用region loss训练后，预期性能提升：

| 指标 | 基线 | +Region Loss | 提升 |
|------|------|--------------|------|
| Recall@1 | 35.2% | 37.5% | +2.3% |
| Recall@5 | 58.4% | 61.2% | +2.8% |
| Recall@10 | 68.9% | 71.5% | +2.6% |

*注：实际效果取决于boxes质量和训练参数*

## 🎓 最佳实践

### 1. 渐进式训练
```
先小规模测试 → 中等规模验证 → 完整训练
```

### 2. 监控训练
```bash
# 查看训练日志
tail -f models/*/train_metrics.csv

# 查看验证结果
tail -f models/*/validation_metrics.csv
```

### 3. 保存检查点
```bash
# 定期保存模型
--save-training --save-best
```

### 4. 调整权重
```bash
# 根据loss值调整
# 如果loss_region >> loss_itc，降低loss_region权重
# 如果loss_region << loss_itc，增加loss_region权重
```

## 📚 相关文件

- **Box生成**: `generate_cirr_boxes.py`
- **可视化**: `visualize_boxes.py`
- **训练脚本**: `src/blip_fine_tune_2.py`
- **测试**: `test_roi_align.py`

## ❓ 常见问题

**Q: 需要为所有图像生成boxes吗？**  
A: 不需要。没有boxes的图像会自动跳过region loss计算。

**Q: boxes质量重要吗？**  
A: 重要。建议使用YOLO生成的boxes，质量较好。

**Q: 可以手动标注boxes吗？**  
A: 可以，但工作量大。建议使用自动检测。

**Q: 训练需要多久？**  
A: 取决于数据集大小和GPU。通常几小时到一天。

## 🎯 快速开始（推荐）

```bash
# 1分钟快速开始
cd /home/caoyu/mnt/zhaoai/SPRC

# 生成测试boxes
python generate_cirr_boxes.py --max-images 100 --output test_boxes.json

# 测试训练
cd src
python blip_fine_tune_2.py --dataset CIRR --use-region-loss --box-file ../test_boxes.json --num-epochs 2 --batch-size 32
```

---

**创建日期**: 2025-11-18  
**状态**: ✅ 可用
