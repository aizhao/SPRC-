#!/bin/bash

# CIRR数据集自动化训练脚本
# 功能：自动生成boxes并训练模型

set -e  # 遇到错误立即退出

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

echo "=========================================="
echo "CIRR数据集自动化训练流程"
echo "=========================================="
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo "数据目录: $DATA_DIR"
echo ""

# 配置参数
CIRR_ROOT="$PROJECT_ROOT/cirr_dataset"
METHOD="yolo"  # yolo, saliency, hybrid
MAX_IMAGES=1000  # 限制图像数量（测试用），设为空则处理全部
BATCH_SIZE=128
NUM_EPOCHS=50
LEARNING_RATE=2e-6

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            echo "🧪 测试模式：只处理100张图像，训练2个epoch"
            MAX_IMAGES=100
            BATCH_SIZE=32
            NUM_EPOCHS=2
            shift
            ;;
        --full)
            echo "🚀 完整模式：处理所有图像"
            MAX_IMAGES=""
            shift
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --help)
            echo "使用方法:"
            echo "  ./auto_train_with_boxes.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --test          测试模式（100张图像，2个epoch）"
            echo "  --full          完整模式（所有图像）"
            echo "  --method METHOD 检测方法（yolo/saliency/hybrid）"
            echo "  --help          显示帮助"
            echo ""
            echo "示例:"
            echo "  ./auto_train_with_boxes.sh --test"
            echo "  ./auto_train_with_boxes.sh --full --method yolo"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 显示配置
echo "配置信息:"
echo "  CIRR根目录: $CIRR_ROOT"
echo "  检测方法: $METHOD"
echo "  图像数量: ${MAX_IMAGES:-全部}"
echo "  Batch Size: $BATCH_SIZE"
echo "  训练轮数: $NUM_EPOCHS"
echo ""

# 步骤1: 检查依赖
echo "步骤1: 检查依赖..."
echo "----------------------------------------"

# 检查Python
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装"
    exit 1
fi
echo "✓ Python已安装"

# 检查CIRR数据集
if [ ! -d "$CIRR_ROOT" ]; then
    echo "❌ CIRR数据集不存在: $CIRR_ROOT"
    echo "请确保数据集路径正确"
    exit 1
fi
echo "✓ CIRR数据集存在"

# 检查YOLO（如果使用）
if [ "$METHOD" = "yolo" ]; then
    if ! python -c "import ultralytics" 2>/dev/null; then
        echo "⚠️  ultralytics未安装，正在安装..."
        pip install ultralytics
    fi
    echo "✓ YOLO已安装"
fi

echo ""

# 步骤2: 生成Bounding Boxes
echo "步骤2: 生成Bounding Boxes..."
echo "----------------------------------------"

# 确保数据目录存在
mkdir -p "$DATA_DIR"

BOX_FILE="$DATA_DIR/cirr_boxes_${METHOD}_${MAX_IMAGES:-full}.json"

if [ -f "$BOX_FILE" ]; then
    echo "⚠️  Box文件已存在: $BOX_FILE"
    read -p "是否重新生成？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过box生成，使用现有文件"
    else
        rm "$BOX_FILE"
        echo "删除旧文件，重新生成..."
    fi
fi

if [ ! -f "$BOX_FILE" ]; then
    MAX_IMAGES_ARG=""
    if [ -n "$MAX_IMAGES" ]; then
        MAX_IMAGES_ARG="--max-images $MAX_IMAGES"
    fi
    
    cd "$PROJECT_ROOT"
    python roi_align_integration/tools/generate_cirr_boxes.py \
        --cirr-root "$CIRR_ROOT" \
        --method "$METHOD" \
        --split train \
        $MAX_IMAGES_ARG \
        --output "$BOX_FILE"
    
    if [ $? -ne 0 ]; then
        echo "❌ Box生成失败"
        exit 1
    fi
fi

echo "✓ Boxes已生成: $BOX_FILE"
echo ""

# 步骤3: 验证Boxes
echo "步骤3: 验证Boxes..."
echo "----------------------------------------"

python -c "
import json
with open('$BOX_FILE', 'r') as f:
    boxes = json.load(f)

total = len(boxes)
with_boxes = sum(1 for b in boxes.values() if len(b) > 0)
total_boxes = sum(len(b) for b in boxes.values())

print(f'✓ 总图像数: {total}')
print(f'✓ 有boxes的图像: {with_boxes} ({with_boxes/total*100:.1f}%)')
print(f'✓ 总box数: {total_boxes}')
print(f'✓ 平均每张: {total_boxes/total:.2f} boxes')

if with_boxes < total * 0.5:
    print('⚠️  警告：超过50%的图像没有boxes')
"

echo ""

# 步骤4: 运行测试
echo "步骤4: 测试功能..."
echo "----------------------------------------"

echo "运行RoI Align功能测试..."
cd "$PROJECT_ROOT"
python roi_align_integration/tests/test_roi_align.py

if [ $? -ne 0 ]; then
    echo "❌ 功能测试失败"
    exit 1
fi

echo "✓ 功能测试通过"
echo ""

# 步骤5: 开始训练
echo "步骤5: 开始训练..."
echo "----------------------------------------"

cd "$PROJECT_ROOT/src"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/roi_align_integration/data/training_${TIMESTAMP}.log"

echo "训练日志将保存到: $LOG_FILE"
echo ""
echo "训练参数:"
echo "  模型: blip2_cir_align_prompt"
echo "  Batch Size: $BATCH_SIZE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  Box文件: $BOX_FILE"
echo ""

read -p "确认开始训练？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消训练"
    exit 0
fi

python blip_fine_tune_2.py \
    --dataset CIRR \
    --blip-model-name blip2_cir_align_prompt \
    --backbone pretrain \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --loss-align 0.4 \
    --loss-rtc 0.4 \
    --use-region-loss \
    --box-file "$BOX_FILE" \
    --loss-region 0.5 \
    --validation-frequency 1 \
    --save-training \
    --save-best \
    2>&1 | tee "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 训练完成！"
    echo "=========================================="
    echo ""
    echo "训练日志: $LOG_FILE"
    echo "模型保存在: models/"
    echo ""
    echo "查看训练指标:"
    echo "  cat models/*/train_metrics.csv"
    echo ""
    echo "查看验证结果:"
    echo "  cat models/*/validation_metrics.csv"
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败"
    echo "=========================================="
    echo ""
    echo "请查看日志: $LOG_FILE"
    exit 1
fi
