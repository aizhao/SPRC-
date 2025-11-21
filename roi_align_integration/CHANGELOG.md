# 更新日志

## v1.1 (2025-11-18) - 文件结构重组

### ✨ 新增
- 创建统一的 `roi_align_integration/` 目录
- 添加 `QUICKSTART.md` 快速开始指南
- 添加 `CHANGELOG.md` 更新日志

### 🔧 修复
- 修复 `target_embeds` 变量名错误（原代码中为 `taregt_embeds`）
- 修复 EVA-CLIP CLS token导致的feature map reshape错误
- 修复混合精度导致的类型不匹配问题（float32 vs float16）
- 更新所有脚本中的路径引用

### 📁 文件结构变更
```
之前：所有文件在根目录
现在：
roi_align_integration/
├── README.md                    # 总览
├── QUICKSTART.md               # 快速开始
├── CHANGELOG.md                # 本文件
├── docs/                        # 文档
│   ├── COMPLETE_SOLUTION.md
│   ├── CIRR_BOX_GENERATION_GUIDE.md
│   ├── ROI_ALIGN_USAGE.md
│   ├── ROI_ALIGN_IMPLEMENTATION_SUMMARY.md
│   └── STEP2_DATA_LOADER_COMPLETE.md
├── tools/                       # 工具
│   ├── auto_train_with_boxes.sh
│   ├── generate_cirr_boxes.py
│   ├── generate_boxes_example.py
│   ├── visualize_boxes.py
│   └── train_with_roi_align.sh
├── tests/                       # 测试
│   └── test_roi_align.py
└── data/                        # 数据
    ├── cirr_boxes_*.json
    └── training_*.log
```

### 🐛 Bug修复详情

#### 1. 变量名拼写错误
**问题**: 在 `blip2_qformer_cir_align_prompt.py` 中，目标图像特征变量名为 `taregt_embeds`（拼写错误），但在调用 `compute_region_loss` 时使用了 `target_embeds`。

**修复**: 统一使用 `taregt_embeds`（虽然是typo，但保持与原代码一致）。

**文件**: `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py:210`

#### 2. Feature Map Reshape错误
**问题**: EVA-CLIP-G的输出包含CLS token（257个tokens），但代码假设是完全平方数（256 = 16×16）。

**错误信息**: `RuntimeError: shape '[16, 16, 1408]' is invalid for input of size 361856`

**修复**: 检测并去除CLS token后再reshape。

```python
# 修复前
feature_map_size = int(image_embeds.shape[1] ** 0.5)
feat_map = image_embeds[i].view(feature_map_size, feature_map_size, hidden_dim)

# 修复后
num_patches = image_embeds.shape[1]
feature_map_size = int((num_patches) ** 0.5)
if feature_map_size * feature_map_size != num_patches:
    feature_map_size = int((num_patches - 1) ** 0.5)
    image_embeds_no_cls = image_embeds[:, 1:, :]  # 去除CLS token
else:
    image_embeds_no_cls = image_embeds
feat_map = image_embeds_no_cls[i].view(feature_map_size, feature_map_size, hidden_dim)
```

**文件**: `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py:316-341`

#### 3. 混合精度类型不匹配
**问题**: 模型在CUDA上使用float16（half precision），但测试数据是float32。

**错误信息**: `RuntimeError: Input type (float) and bias type (c10::Half) should be the same`

**修复**: 在测试脚本中将模型转换为float32。

```python
def load_test_model(device="cuda"):
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_cir_align_prompt", 
        model_type="pretrain", 
        is_eval=False, 
        device=device
    )
    # 将模型转换为float32以避免类型不匹配
    model = model.float()
    return model, vis_processors, txt_processors
```

**文件**: `roi_align_integration/tests/test_roi_align.py:20-27`

### 📊 测试结果

修复后的测试结果：
- ✅ 测试1: 基本前向传播 - 通过
- ✅ 测试2: 区域损失功能 - 通过
- ✅ 测试3: 区域特征提取 - 通过
- ✅ 测试4: 空boxes处理 - 通过
- ✅ 测试5: 不同数量boxes - 通过

**通过率**: 5/5 (100%) ✨

---

## v1.0 (2025-11-18) - 初始版本

### ✨ 功能
- 集成FG-CLIP的RoI Align功能到SPRC
- 实现区域级对比损失（region-level contrastive loss）
- 添加bounding box数据支持
- 创建Box生成工具
- 编写完整文档和测试

### 📝 核心修改
1. **模型修改** (`blip2_qformer_cir_align_prompt.py`)
   - 添加 `extract_region_features()` 方法
   - 添加 `compute_region_loss()` 方法
   - 集成region loss到forward方法

2. **训练脚本修改** (`blip_fine_tune_2.py`)
   - 添加 `--use-region-loss` 参数
   - 添加 `--loss-region` 参数
   - 添加 `--box-file` 参数
   - 支持可变长度batch数据

3. **数据加载器修改** (`data_utils.py`)
   - CIRRDataset支持加载boxes
   - 返回ref_boxes和tgt_boxes

### 📚 文档
- ROI_ALIGN_USAGE.md - 使用文档
- ROI_ALIGN_IMPLEMENTATION_SUMMARY.md - 实现总结
- CIRR_BOX_GENERATION_GUIDE.md - Box生成指南
- STEP2_DATA_LOADER_COMPLETE.md - 数据加载器说明
- COMPLETE_SOLUTION.md - 完整解决方案

### 🛠️ 工具
- generate_cirr_boxes.py - CIRR Box生成工具
- generate_boxes_example.py - Box生成示例
- visualize_boxes.py - Box可视化工具
- auto_train_with_boxes.sh - 自动化训练脚本
- train_with_roi_align.sh - 训练脚本示例

### 🧪 测试
- test_roi_align.py - 功能测试套件

---

## 未来计划

### v1.2 (计划中)
- [ ] 添加更多检测方法（Faster R-CNN, SAM等）
- [ ] 支持多尺度RoI Align
- [ ] 添加Box质量评估工具
- [ ] 优化内存使用

### v1.3 (计划中)
- [ ] 支持其他数据集（FashionIQ等）
- [ ] 添加可视化训练过程的工具
- [ ] 提供预训练的Box文件下载

---

**维护者**: Cascade AI Assistant  
**最后更新**: 2025-11-18
