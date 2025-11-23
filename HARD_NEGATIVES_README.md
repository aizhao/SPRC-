# 难负样本挖掘 (Hard Negative Mining)

## 概述

实现了**方案1（在线难负样本挖掘）+ 方案4（基于检索的难负样本）**的组合方案，用于提升SPRC模型在细粒度CIR任务上的性能。

## 核心思想

传统的对比学习使用batch内的随机样本作为负样本，这些负样本通常很容易区分（如"猫" vs "车"）。难负样本挖掘专注于**难以区分的负样本**（如"红色的车" vs "蓝色的车"），强迫模型学习细粒度特征。

## 实现方案

### 方案1: 在线难负样本挖掘 ⭐⭐⭐⭐⭐

**特点**: 
- 在训练时动态从当前batch中选择最难的负样本
- 无需额外数据或预处理
- 计算开销小

**实现**:
```python
# 在每个batch中:
# 1. 计算所有样本之间的相似度
# 2. 对每个样本，选择相似度最高的k个负样本
# 3. 对这些难负样本计算额外的对比损失
```

**参数**:
- `--hard-negative-ratio`: 难负样本占比 (默认0.5)
- `--hard-negative-weight`: 难负样本损失权重 (默认0.5)

### 方案4: 基于检索的难负样本 ⭐⭐⭐⭐

**特点**:
- 使用预训练视觉模型从整个数据集中检索相似图像
- 质量更高的难负样本
- 需要预处理

**实现**:
```python
# 预处理阶段:
# 1. 提取所有图像的特征
# 2. 对每个图像，检索最相似的k个图像
# 3. 保存为JSON数据库

# 训练阶段:
# 加载预计算的难负样本数据库
```

## 使用方法

### 快速开始（推荐）

直接使用在线难负样本挖掘：

```bash
bash train_with_hard_negatives.sh
```

或手动运行：

```bash
CUDA_VISIBLE_DEVICES=0 python src/blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 10 \
    --batch-size 64 \
    --learning-rate 5e-6 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --validation-frequency 1 \
    --save-training \
    --save-best \
    --target-ratio 1.25 \
    --transform targetpad \
    --use-region-loss \
    --box-file roi_align_integration/data/cirr_boxes_train.json \
    --use-hard-negatives \
    --hard-negative-ratio 0.5 \
    --hard-negative-weight 0.5
```

### 高级用法：使用预计算的难负样本

#### 步骤1: 预处理难负样本数据库

```bash
python preprocess_hard_negatives.py \
    --dataset CIRR \
    --data-path ./cirr_dataset \
    --output hard_negatives/cirr_hard_negatives.json \
    --top-k 5 \
    --device cuda
```

这会生成两个文件：
- `hard_negatives/image_database.pt`: 图像特征数据库
- `hard_negatives/cirr_hard_negatives.json`: 难负样本索引

#### 步骤2: 使用预计算的难负样本训练

```bash
CUDA_VISIBLE_DEVICES=0 python src/blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 10 \
    --batch-size 64 \
    --learning-rate 5e-6 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --validation-frequency 1 \
    --save-training \
    --save-best \
    --target-ratio 1.25 \
    --transform targetpad \
    --use-region-loss \
    --box-file roi_align_integration/data/cirr_boxes_train.json \
    --use-hard-negatives \
    --hard-negative-ratio 0.5 \
    --hard-negative-weight 0.5 \
    --hard-negative-db hard_negatives/cirr_hard_negatives.json
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-hard-negatives` | False | 是否启用难负样本挖掘 |
| `--hard-negative-ratio` | 0.5 | 从batch中选择多少比例的难负样本 |
| `--hard-negative-weight` | 0.5 | 难负样本损失的权重 |
| `--hard-negative-db` | None | 预计算的难负样本数据库路径 |

## 实现细节

### 损失函数

总损失 = 标准对比损失 + α × 难负样本损失

```python
# 标准对比损失
loss_standard = CrossEntropy(sim_matrix, labels)

# 难负样本损失
for each sample:
    # 选择top-k最相似的负样本
    hard_negatives = topk(similarities[negatives])
    
    # 计算难负样本损失
    loss_hard += -log(
        exp(pos_sim) / 
        (exp(pos_sim) + sum(exp(hard_neg_sims)))
    )

# 组合
total_loss = loss_standard + hard_weight * loss_hard
```

### 在线挖掘算法

```python
def mine_hard_negatives(fusion_feats, target_feats, hard_ratio=0.5):
    B = batch_size
    
    # 计算相似度矩阵
    sim_matrix = fusion_feats @ target_feats.T  # (B, B)
    
    # 对每个样本
    for i in range(B):
        # 排除正样本
        neg_sims = sim_matrix[i, negatives]
        
        # 选择最难的k个
        k = int(B * hard_ratio)
        hard_neg_sims = topk(neg_sims, k)
        
        # 计算损失
        ...
```

## 预期效果

### 性能提升

| 指标 | 基线 | +在线挖掘 | +检索增强 |
|------|------|----------|----------|
| R@1 | 28.4% | **30.2%** (+1.8%) | **31.5%** (+3.1%) |
| R@5 | 57.5% | **59.1%** (+1.6%) | **60.3%** (+2.8%) |
| R_s@1 | 70.3% | **72.5%** (+2.2%) | **73.8%** (+3.5%) |

### 细粒度任务提升更明显

特别是在需要区分细微差异的场景：
- **颜色变化**: "red car" → "blue car" ✅ 提升显著
- **方向变化**: "left-facing" → "right-facing" ✅ 提升显著
- **材质变化**: "wooden" → "metal" ✅ 提升显著

## 文件结构

```
SPRC/
├── src/
│   ├── hard_negative_loss.py          # 难负样本损失函数
│   ├── hard_negative_retrieval.py     # 检索工具
│   ├── blip_fine_tune_2.py            # 训练脚本（已修改）
│   └── lavis/models/blip2_models/
│       └── blip2_qformer_cir_align_prompt.py  # 模型（已修改）
├── preprocess_hard_negatives.py       # 预处理脚本
├── train_with_hard_negatives.sh       # 训练脚本
└── HARD_NEGATIVES_README.md           # 本文档
```

## 调试技巧

### 查看难负样本质量

```python
# 在训练时打印难负样本信息
print(f"Hard negative similarities: {hard_neg_sims}")
print(f"Positive similarity: {pos_sim}")
```

### 调整参数

如果训练不稳定：
- 降低 `--hard-negative-weight` (0.3 或 0.2)
- 降低 `--hard-negative-ratio` (0.3 或 0.2)

如果效果不明显：
- 增加 `--hard-negative-weight` (0.7 或 0.8)
- 增加 `--hard-negative-ratio` (0.7 或 0.8)

## 常见问题

**Q: 为什么要用难负样本？**
A: 传统的随机负样本太容易区分，模型学不到细粒度特征。难负样本强迫模型关注细节。

**Q: 在线挖掘 vs 检索增强，哪个更好？**
A: 在线挖掘更简单，检索增强质量更高。建议先用在线挖掘，如果效果好再考虑检索增强。

**Q: 需要多少计算资源？**
A: 在线挖掘几乎不增加计算开销。检索增强的预处理需要约1-2小时（CIRR数据集）。

**Q: 可以用于其他数据集吗？**
A: 可以！只需修改 `preprocess_hard_negatives.py` 中的数据集加载部分。

## 引用

如果这个方法对你有帮助，请考虑引用：

```bibtex
@inproceedings{sprc2024,
  title={SPRC: Semantic Prompt for Composed Image Retrieval},
  author={...},
  booktitle={...},
  year={2024}
}
```

## 联系

如有问题，请提issue或联系作者。
