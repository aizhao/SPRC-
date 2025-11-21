# RoI Align区域特征提取模块使用指南

## 📋 概述

本文档说明如何使用新添加的RoI Align区域特征提取功能，该功能从FG-CLIP移植而来，用于提升CIRR数据集上的细粒度检索性能。

## 🎯 功能说明

### 核心功能
- **区域特征提取**: 使用RoI Align从图像的密集特征图中提取指定区域的特征
- **区域级对比损失**: 计算参考图像和目标图像对应区域之间的对比损失
- **灵活控制**: 可以通过命令行参数启用/禁用该功能

### 实现位置
- **模型文件**: `/home/caoyu/mnt/zhaoai/SPRC/src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py`
- **训练脚本**: `/home/caoyu/mnt/zhaoai/SPRC/src/blip_fine_tune_2.py`

## 🚀 使用方法

### 方法1: 不使用区域损失（默认）

```bash
cd /home/caoyu/mnt/zhaoai/SPRC/src

python blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs 50 \
    --batch-size 128 \
    --learning-rate 2e-6 \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --save-training \
    --save-best
```

### 方法2: 启用区域损失（需要bounding box数据）

```bash
cd /home/caoyu/mnt/zhaoai/SPRC/src

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
    --loss-region 0.5 \
    --save-training \
    --save-best
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-region-loss` | flag | False | 是否启用RoI Align区域损失 |
| `--loss-region` | float | 0.5 | 区域损失的权重 |
| `--loss-align` | float | 0.4 | 对齐损失的权重 |
| `--loss-rtc` | float | 0.4 | 相对对比损失的权重 |

## 📝 代码集成说明

### 1. 模型修改

在 `blip2_qformer_cir_align_prompt.py` 中添加了以下功能：

#### a) RoI Align特征提取
```python
def extract_region_features(self, image_embeds, boxes, image_size=(224, 224)):
    """
    使用RoI Align从图像特征中提取区域特征
    
    Args:
        image_embeds: 图像特征 (B, N, D) 其中N是patch数量
        boxes: bounding boxes列表，每个元素是该图像的boxes (x1, y1, x2, y2)格式，归一化到[0,1]
    
    Returns:
        region_features: 区域特征列表
    """
```

#### b) 区域级对比损失
```python
def compute_region_loss(self, ref_image_embeds, target_image_embeds, ref_boxes, target_boxes):
    """
    计算区域级对比损失
    
    Args:
        ref_image_embeds: 参考图像特征
        target_image_embeds: 目标图像特征
        ref_boxes: 参考图像的bounding boxes
        target_boxes: 目标图像的bounding boxes
    
    Returns:
        loss_region: 区域级对比损失
    """
```

### 2. 前向传播修改

在 `forward` 方法中：
```python
def forward(self, samples):
    # ... 原有的损失计算 ...
    
    losses = {
        'loss_itc': loss_itc, 
        'loss_rtc': loss_rtc,
        'loss_align': loss_align
    }
    
    # 如果提供了region boxes，计算区域级损失
    if self.use_region_loss and 'region_boxes' in samples and samples['region_boxes'] is not None:
        loss_region = self.compute_region_loss(
            image_embeds, target_embeds, 
            samples['region_boxes'], samples.get('target_region_boxes')
        )
        losses['loss_region'] = loss_region
    
    return losses
```

## 🔧 如何添加Bounding Box数据

### 当前状态
目前代码中 `region_boxes` 设置为 `None`，因此区域损失不会被计算。

### 添加Box数据的步骤

#### 步骤1: 准备Box标注数据

创建一个JSON文件，格式如下：
```json
{
    "image_name_1": [
        [x1, y1, x2, y2],  // 第一个box，归一化到[0,1]
        [x1, y1, x2, y2]   // 第二个box
    ],
    "image_name_2": [
        [x1, y1, x2, y2]
    ]
}
```

#### 步骤2: 修改数据加载器

在 `data_utils.py` 的 `CIRRDataset` 类中添加box加载：

```python
class CIRRDataset(Dataset):
    def __init__(self, split: str, mode: str, preprocess: callable, box_file: str = None):
        # ... 原有代码 ...
        
        # 加载box数据
        self.boxes = {}
        if box_file and os.path.exists(box_file):
            with open(box_file, 'r') as f:
                self.boxes = json.load(f)
    
    def __getitem__(self, index):
        if self.mode == 'relative' and self.split == 'train':
            # ... 原有代码 ...
            
            # 获取boxes
            ref_boxes = self.boxes.get(reference_name, [])
            tgt_boxes = self.boxes.get(target_hard_name, [])
            
            return reference_image, target_image, rel_caption, ref_boxes, tgt_boxes
```

#### 步骤3: 修改训练循环

在 `blip_fine_tune_2.py` 中：

```python
for idx, batch_data in enumerate(train_bar):
    if len(batch_data) == 5:
        reference_images, target_images, captions, ref_boxes, tgt_boxes = batch_data
    else:
        reference_images, target_images, captions = batch_data
        ref_boxes, tgt_boxes = None, None
    
    # ... 其他代码 ...
    
    loss_dict = blip_model({
        "image": reference_images, 
        "target": target_images, 
        "text_input": captions,
        "region_boxes": ref_boxes,
        "target_region_boxes": tgt_boxes
    })
```

## 📊 预期效果

### 性能提升
- **细粒度检索**: 通过关注局部区域，提升对细微差异的识别能力
- **属性理解**: 更好地理解颜色、形状等局部属性
- **空间关系**: 改善对物体位置和空间关系的理解

### 适用场景
- CIRR数据集的训练和评估
- 需要细粒度理解的图像检索任务
- 有bounding box标注的数据集

## ⚠️ 注意事项

### 1. 内存消耗
- RoI Align会增加一定的内存消耗
- 建议根据GPU内存调整batch size

### 2. Box数据格式
- Boxes必须归一化到[0, 1]范围
- 格式为 `[x1, y1, x2, y2]`，其中 `(x1, y1)` 是左上角，`(x2, y2)` 是右下角

### 3. 训练时间
- 启用区域损失会略微增加训练时间（约10-15%）

### 4. 渐进式训练策略
建议采用两阶段训练：
1. **第一阶段**: 不使用区域损失，训练全局特征（20-30 epochs）
2. **第二阶段**: 启用区域损失，微调细粒度特征（10-20 epochs）

## 🧪 测试代码

创建一个简单的测试脚本验证功能：

```python
import torch
from lavis.models import load_model_and_preprocess

# 加载模型
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_cir_align_prompt", 
    model_type="pretrain", 
    is_eval=False, 
    device="cuda"
)

# 启用区域损失
model.use_region_loss = True

# 创建测试数据
batch_size = 2
image = torch.randn(batch_size, 3, 224, 224).cuda()
target = torch.randn(batch_size, 3, 224, 224).cuda()
text = ["a red car", "a blue shirt"]

# 测试boxes（归一化坐标）
ref_boxes = [
    [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]],  # 第一张图的2个boxes
    [[0.2, 0.2, 0.8, 0.8]]  # 第二张图的1个box
]
tgt_boxes = [
    [[0.15, 0.15, 0.55, 0.55], [0.65, 0.65, 0.95, 0.95]],
    [[0.25, 0.25, 0.85, 0.85]]
]

# 前向传播
samples = {
    "image": image,
    "target": target,
    "text_input": text,
    "region_boxes": ref_boxes,
    "target_region_boxes": tgt_boxes
}

losses = model(samples)
print("Losses:", losses)
```

## 📚 参考资料

- **FG-CLIP论文**: [FG-CLIP: Fine-Grained Visual and Textual Alignment](https://arxiv.org/abs/2505.05071)
- **RoI Align**: [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- **CIRR数据集**: [Composed Image Retrieval using Contrastive Learning](https://arxiv.org/abs/2104.03015)

## 🤝 贡献

如果你发现bug或有改进建议，请：
1. 检查代码实现
2. 运行测试
3. 提交issue或pull request

## 📞 联系方式

如有问题，请查看：
- 代码注释
- 本文档
- FG-CLIP原始实现
