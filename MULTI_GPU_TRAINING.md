# 多GPU训练指南

## 修改说明

已为训练脚本添加了多GPU支持，使用 `torch.nn.DataParallel` 实现。

### 修改的文件

1. **src/blip_fine_tune_2.py**
   - 在 `clip_finetune_cirr()` 函数中添加了多GPU支持
   - 在 `clip_finetune_fiq()` 函数中添加了多GPU支持
   - 自动检测可用GPU数量并使用DataParallel包装模型

2. **src/utils.py**
   - 修改了 `save_model()` 函数以正确处理DataParallel包装的模型
   - 修改了 `collate_fn()` 函数以处理可变长度的bounding boxes

3. **src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py**
   - 修复了 `region_proj` 层的维度问题

## 使用方法

### 1. 检查可用GPU

```bash
python check_gpus.py
```

### 2. 指定使用的GPU

如果你想使用特定的GPU（例如GPU 0和1），可以通过环境变量指定：

```bash
# 使用GPU 0和1
export CUDA_VISIBLE_DEVICES=0,1

# 或者在命令前直接指定
CUDA_VISIBLE_DEVICES=0,1 python src/blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 1 \
    --batch-size 64 \
    --learning-rate 1e-5 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --use-region-loss \
    --box-file roi_align_integration/data/cirr_boxes_train.json \
    --loss-region 0.5 \
    --validation-frequency 1 \
    --save-training \
    --save-best \
    --target-ratio 1.25
```

### 3. 自动使用所有可用GPU

如果不设置 `CUDA_VISIBLE_DEVICES`，程序会自动使用所有可用的GPU：

```bash
python src/blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 1 \
    --batch-size 64 \
    --learning-rate 1e-5 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --use-region-loss \
    --box-file roi_align_integration/data/cirr_boxes_train.json \
    --loss-region 0.5 \
    --validation-frequency 1 \
    --save-training \
    --save-best \
    --target-ratio 1.25
```

## 注意事项

1. **Batch Size**: 使用多GPU时，实际的batch size会在多个GPU上分配。例如，如果设置 `--batch-size 64` 并使用2个GPU，每个GPU会处理32个样本。

2. **内存使用**: 确保每个GPU都有足够的内存。如果遇到OOM错误，可以减小batch size。

3. **性能**: DataParallel会在每次前向传播时复制模型到所有GPU，这可能会有一些开销。对于更大规模的训练，建议使用 `DistributedDataParallel`。

4. **模型保存**: 保存的模型权重已经去除了DataParallel的包装，可以直接在单GPU或多GPU环境中加载。

## 预期输出

训练开始时会显示：

```
Using 2 GPUs for training
```

或

```
Using single GPU for training
```

## 故障排除

### 问题：显示"Using single GPU for training"但系统有多个GPU

**解决方案**：
- 检查 `CUDA_VISIBLE_DEVICES` 环境变量
- 运行 `python check_gpus.py` 确认PyTorch能看到多个GPU
- 确保没有其他进程占用GPU

### 问题：OOM (Out of Memory) 错误

**解决方案**：
- 减小 `--batch-size` 参数
- 减少 `--num-workers` 参数
- 使用更少的GPU（通过 `CUDA_VISIBLE_DEVICES` 指定）

### 问题：训练速度没有明显提升

**解决方案**：
- 确保batch size足够大，以充分利用多GPU
- 检查是否有数据加载瓶颈（可以增加 `--num-workers`）
- 考虑使用DistributedDataParallel代替DataParallel
