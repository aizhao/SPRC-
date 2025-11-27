# 文本增强方案 (Text Enhancement for SmartCLIP-SPRC)

## 背景

在CIR任务中集成SmartCLIP的MaskNetwork时，存在核心矛盾：
- **修改文本描述目标图像**（例如："change to blue"）
- **但mask作用在参考图像上**（红色连衣裙）
- 参考图像没有"蓝色"信息可供mask激活

## 解决方案：文本增强

将参考图像的caption与修改文本组合：
```
原始文本: "change to blue"
增强文本: "A red dress. change to blue"
```

这样mask可以：
1. 理解参考图像的内容（"red dress"）
2. 理解需要的修改（"change to blue"）
3. 选择相关的特征维度

## 使用步骤

### 1. 生成参考图像的captions

```bash
# 为训练集生成captions
python generate_reference_captions.py --split train

# 为验证集生成captions
python generate_reference_captions.py --split val
```

这将生成文件：
- `datasets/CIRR/cirr/captions/reference_captions.train.json`
- `datasets/CIRR/cirr/captions/reference_captions.val.json`

### 2. 训练时启用文本增强

修改训练脚本，在创建数据集时添加`use_enhanced_text=True`：

```python
# 原始代码
relative_train_dataset = CIRRDataset('train', 'relative', preprocess)

# 使用文本增强
relative_train_dataset = CIRRDataset('train', 'relative', preprocess, use_enhanced_text=True)
```

### 3. 验证时也启用文本增强

```python
relative_val_dataset = CIRRDataset('val', 'relative', preprocess, use_enhanced_text=True)
```

## 文本格式

增强后的文本格式为：
```
"{reference_caption}. {modification_text}"
```

示例：
- 参考图像caption: "A woman wearing a red dress"
- 修改文本: "change the dress to blue"
- 增强文本: "A woman wearing a red dress. change the dress to blue"

## 预期效果

1. **Mask更有意义**：
   - 文本描述"red dress" → mask可以定位服装相关维度
   - 文本描述"change to blue" → mask知道需要修改颜色维度

2. **符合SmartCLIP设计**：
   - SmartCLIP的文本描述图像内容
   - 现在CIR的文本也描述参考图像内容

3. **性能提升**：
   - Mask有更多上下文信息
   - 可以更准确地选择相关特征

## 注意事项

1. **Caption质量**：使用BLIP2生成的caption质量较高，但可能不完美
2. **文本长度**：增强后文本更长，确保`max_txt_len`足够（建议≥64）
3. **推理一致性**：训练和推理都要使用相同的文本增强策略

## 文件修改清单

- ✅ `generate_reference_captions.py` - Caption生成脚本
- ✅ `src/data_utils.py` - 数据集类添加`use_enhanced_text`参数
- ⏳ 训练脚本 - 需要手动添加`use_enhanced_text=True`
