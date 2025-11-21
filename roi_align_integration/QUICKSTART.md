# 🚀 快速开始指南

## 一分钟上手

```bash
# 1. 进入项目目录
cd /home/caoyu/mnt/zhaoai/SPRC

# 2. 给脚本执行权限
chmod +x roi_align_integration/tools/auto_train_with_boxes.sh

# 3. 运行测试（100张图像，2个epoch）
roi_align_integration/tools/auto_train_with_boxes.sh --test
```

就这么简单！✨

## 测试成功后

如果测试通过，运行完整训练：

```bash
roi_align_integration/tools/auto_train_with_boxes.sh --full
```

## 文件位置

所有生成的文件都在 `roi_align_integration/` 目录下：

```
roi_align_integration/
├── data/                    # 📦 生成的数据
│   ├── cirr_boxes_*.json   # Bounding box文件
│   └── training_*.log      # 训练日志
├── docs/                    # 📚 详细文档
├── tools/                   # 🛠️ 工具脚本
└── tests/                   # 🧪 测试脚本
```

## 需要帮助？

查看完整文档：
```bash
cat roi_align_integration/docs/COMPLETE_SOLUTION.md
```

或查看README：
```bash
cat roi_align_integration/README.md
```

## 常见命令

```bash
# 使用不同的检测方法
roi_align_integration/tools/auto_train_with_boxes.sh --test --method saliency

# 查看帮助
roi_align_integration/tools/auto_train_with_boxes.sh --help

# 手动生成boxes
python roi_align_integration/tools/generate_cirr_boxes.py \
    --cirr-root ./cirr_dataset \
    --method yolo \
    --output roi_align_integration/data/my_boxes.json

# 运行测试
python roi_align_integration/tests/test_roi_align.py

# 可视化boxes
python roi_align_integration/tools/visualize_boxes.py \
    --box-file roi_align_integration/data/cirr_boxes_yolo_100.json \
    --image-dir ./cirr_dataset/dev
```

## 故障排除

### 问题：测试失败

1. 检查CUDA是否可用：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. 查看详细错误：
```bash
python roi_align_integration/tests/test_roi_align.py 2>&1 | less
```

### 问题：Box生成太慢

使用更快的方法：
```bash
roi_align_integration/tools/auto_train_with_boxes.sh --test --method saliency
```

### 问题：内存不足

减小batch size：
编辑脚本或手动训练时使用 `--batch-size 64`

---

**开始使用**: `roi_align_integration/tools/auto_train_with_boxes.sh --test` 🎯
