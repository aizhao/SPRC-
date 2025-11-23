"""
预处理脚本：为CIRR数据集生成难负样本数据库

使用方法:
python preprocess_hard_negatives.py --dataset CIRR --output hard_negatives/cirr_hard_negatives.json
"""
import argparse
import torch
from pathlib import Path
import sys
sys.path.append('src')

from hard_negative_retrieval import HardNegativeRetriever
from lavis.models import load_model_and_preprocess
from data_utils import CIRRDataset, targetpad_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIRR", help="Dataset name")
    parser.add_argument("--data-path", type=str, default="./cirr_dataset", help="Path to dataset")
    parser.add_argument("--output", type=str, required=True, help="Output path for hard negatives database")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hard negatives per sample")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hard Negative Preprocessing for CIR")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. Loading visual encoder...")
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_cir_align_prompt",
        model_type="pretrain",
        is_eval=True,
        device=args.device
    )
    
    # 使用完整的visual encoder + ln_vision pipeline
    class VisualEncoderWrapper(torch.nn.Module):
        def __init__(self, visual_encoder, ln_vision):
            super().__init__()
            self.visual_encoder = visual_encoder
            self.ln_vision = ln_vision
            # 检测模型的dtype
            self.model_dtype = next(visual_encoder.parameters()).dtype
            print(f"Visual encoder dtype: {self.model_dtype}")
            self.eval()
        
        def forward(self, x):
            # 转换输入到模型的dtype (明确使用half()或float())
            if self.model_dtype == torch.float16:
                x = x.half()
            elif self.model_dtype == torch.float32:
                x = x.float()
            # 使用与训练时相同的pipeline
            return self.ln_vision(self.visual_encoder(x))
    
    visual_encoder = VisualEncoderWrapper(blip_model.visual_encoder, blip_model.ln_vision)
    visual_encoder.eval()
    
    # 将整个模型转换为float32以避免混合精度问题
    visual_encoder = visual_encoder.float()
    print("Converted visual encoder to float32")
    
    # 2. 加载数据集
    print("\n2. Loading dataset...")
    preprocess = targetpad_transform(1.25, 224)
    
    if args.dataset.upper() == "CIRR":
        dataset = CIRRDataset(
            split='train',
            mode='relative',
            preprocess=preprocess
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    print(f"Dataset size: {len(dataset)}")
    
    # 3. 创建检索器
    print("\n3. Creating hard negative retriever...")
    retriever = HardNegativeRetriever(
        visual_encoder=visual_encoder,
        device=args.device,
        top_k=args.top_k
    )
    
    # 4. 预处理难负样本
    print("\n4. Preprocessing hard negatives...")
    print(f"This may take a while...")
    
    hard_negatives_db = retriever.preprocess_dataset(
        dataset=dataset,
        save_path=args.output,
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing completed!")
    print(f"✓ Hard negatives database saved to: {args.output}")
    print(f"✓ Total samples: {len(hard_negatives_db)}")
    print(f"✓ Hard negatives per sample: {args.top_k}")
    print("=" * 60)
    
    # 5. 显示一些统计信息
    print("\nSample statistics:")
    sample_idx = 0
    if sample_idx in hard_negatives_db:
        sample_data = hard_negatives_db[sample_idx]
        print(f"Sample {sample_idx}:")
        print(f"  - Hard negative indices: {sample_data['indices']}")
        print(f"  - Similarity scores: {[f'{s:.4f}' for s in sample_data['scores']]}")


if __name__ == "__main__":
    main()
