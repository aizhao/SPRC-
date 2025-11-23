"""
基于检索的难负样本生成
使用CLIP等预训练模型检索相似但不同的图像
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
from pathlib import Path
import numpy as np


class HardNegativeRetriever:
    """
    难负样本检索器
    
    使用预训练的视觉模型（如CLIP）从图像库中检索难负样本
    """
    def __init__(self, visual_encoder, device='cuda', top_k=5):
        """
        Args:
            visual_encoder: 视觉编码器（如CLIP的image encoder）
            device: 设备
            top_k: 每个样本检索k个难负样本
        """
        self.visual_encoder = visual_encoder
        self.device = device
        self.top_k = top_k
        self.visual_encoder.eval()
    
    @torch.no_grad()
    def build_image_database(self, dataset, save_path=None):
        """
        构建图像特征数据库
        
        Args:
            dataset: 数据集
            save_path: 保存路径
        
        Returns:
            features: (N, D) - 所有图像的特征
            image_paths: List[str] - 图像路径
        """
        print("Building image feature database...")
        
        features_list = []
        image_paths = []
        
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            
            # 处理不同的数据格式
            if isinstance(sample, tuple):
                # CIRRDataset在relative模式下返回tuple: (ref_img, tgt_img, caption)
                if len(sample) >= 2:
                    target_image = sample[1]  # 第二个是target_image
                else:
                    target_image = sample[0]
            elif isinstance(sample, dict):
                target_image = sample['target']
            else:
                target_image = sample
            
            # 确保是4D tensor (B, C, H, W)
            if target_image.dim() == 3:
                target_image = target_image.unsqueeze(0)
            
            # 移动到设备（dtype由模型自动处理）
            target_image = target_image.to(self.device)
            
            # 提取特征
            feat = self.visual_encoder(target_image)
            if isinstance(feat, tuple):
                feat = feat[0]  # 如果返回多个值，取第一个
            
            # 全局池化（如果是多个token）
            if len(feat.shape) == 3:  # (1, N, D)
                feat = feat[:, 0, :]  # 取CLS token
            
            features_list.append(feat.cpu())
            image_paths.append(f'image_{idx}')
        
        features = torch.cat(features_list, dim=0)  # (N, D)
        features = F.normalize(features, dim=-1)  # L2归一化
        
        # 保存
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'features': features,
                'image_paths': image_paths
            }, save_path)
            print(f"Database saved to {save_path}")
        
        return features, image_paths
    
    @torch.no_grad()
    def retrieve_hard_negatives(self, target_image, database_features, 
                                exclude_indices=None, return_scores=False):
        """
        为单个目标图像检索难负样本
        
        Args:
            target_image: (1, 3, H, W) - 目标图像
            database_features: (N, D) - 数据库特征
            exclude_indices: List[int] - 要排除的索引（如自己）
            return_scores: 是否返回相似度分数
        
        Returns:
            hard_neg_indices: (top_k,) - 难负样本的索引
            scores: (top_k,) - 相似度分数（可选）
        """
        # 提取目标图像特征（dtype由模型自动处理）
        target_image = target_image.to(self.device)
        
        feat = self.visual_encoder(target_image)
        if isinstance(feat, tuple):
            feat = feat[0]
        if len(feat.shape) == 3:
            feat = feat[:, 0, :]
        feat = F.normalize(feat, dim=-1)
        
        # 计算相似度
        similarities = torch.matmul(feat, database_features.t()).squeeze()  # (N,)
        
        # 排除指定索引
        if exclude_indices is not None:
            similarities[exclude_indices] = -float('inf')
        
        # 选择top-k最相似的（但不是自己）
        topk_scores, topk_indices = similarities.topk(self.top_k + 10)  # 多取一些以防重复
        
        # 过滤掉完全相同的图像（相似度 > 0.99）
        valid_mask = topk_scores < 0.99
        topk_indices = topk_indices[valid_mask][:self.top_k]
        topk_scores = topk_scores[valid_mask][:self.top_k]
        
        if return_scores:
            return topk_indices.cpu(), topk_scores.cpu()
        return topk_indices.cpu()
    
    def preprocess_dataset(self, dataset, save_path, batch_size=32):
        """
        为整个数据集预处理难负样本
        
        Args:
            dataset: 数据集
            save_path: 保存路径
            batch_size: 批次大小
        
        Returns:
            hard_negatives_db: Dict[int, List[int]] - 每个样本的难负样本索引
        """
        print("Preprocessing hard negatives for dataset...")
        
        # 1. 构建特征数据库
        db_path = Path(save_path).parent / 'image_database.pt'
        if db_path.exists():
            print(f"Loading existing database from {db_path}")
            db_data = torch.load(db_path)
            database_features = db_data['features']
            image_paths = db_data['image_paths']
        else:
            database_features, image_paths = self.build_image_database(
                dataset, save_path=db_path
            )
        
        database_features = database_features.to(self.device)
        
        # 2. 为每个样本检索难负样本
        hard_negatives_db = {}
        
        for idx in tqdm(range(len(dataset)), desc="Retrieving hard negatives"):
            sample = dataset[idx]
            
            # 处理不同的数据格式
            if isinstance(sample, tuple):
                # CIRRDataset在relative模式下返回tuple: (ref_img, tgt_img, caption)
                if len(sample) >= 2:
                    target_image = sample[1]  # 第二个是target_image
                else:
                    target_image = sample[0]
            elif isinstance(sample, dict):
                target_image = sample['target']
            else:
                target_image = sample
            
            # 确保是4D tensor
            if target_image.dim() == 3:
                target_image = target_image.unsqueeze(0)
            
            # 检索难负样本（排除自己）
            hard_neg_indices, scores = self.retrieve_hard_negatives(
                target_image,
                database_features,
                exclude_indices=[idx],
                return_scores=True
            )
            
            hard_negatives_db[idx] = {
                'indices': hard_neg_indices.tolist(),
                'scores': scores.tolist()
            }
        
        # 3. 保存
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(hard_negatives_db, f, indent=2)
        
        print(f"Hard negatives database saved to {save_path}")
        print(f"Total samples: {len(hard_negatives_db)}")
        print(f"Hard negatives per sample: {self.top_k}")
        
        return hard_negatives_db


def load_hard_negatives_db(db_path):
    """加载预处理的难负样本数据库"""
    with open(db_path, 'r') as f:
        db = json.load(f)
    
    # 转换键为整数
    db = {int(k): v for k, v in db.items()}
    return db


def get_hard_negative_samples(dataset, indices, hard_negatives_db, num_samples=1):
    """
    从数据集中获取难负样本
    
    Args:
        dataset: 数据集
        indices: 当前batch的索引
        hard_negatives_db: 难负样本数据库
        num_samples: 每个样本取几个难负样本
    
    Returns:
        hard_negative_images: List[Tensor] - 难负样本图像
    """
    hard_negative_images = []
    
    for idx in indices:
        if idx in hard_negatives_db:
            hard_neg_indices = hard_negatives_db[idx]['indices']
            
            # 随机选择num_samples个
            selected_indices = np.random.choice(
                hard_neg_indices, 
                size=min(num_samples, len(hard_neg_indices)),
                replace=False
            )
            
            # 获取图像
            for neg_idx in selected_indices:
                neg_sample = dataset[neg_idx]
                hard_negative_images.append(neg_sample['target'])
        else:
            # 如果没有预处理的难负样本，随机选择
            random_idx = np.random.randint(0, len(dataset))
            hard_negative_images.append(dataset[random_idx]['target'])
    
    return hard_negative_images
