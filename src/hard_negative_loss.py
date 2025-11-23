"""
Hard Negative Mining for CIR
在线难负样本挖掘 + 检索增强的组合方案
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeContrastiveLoss(nn.Module):
    """
    难负样本对比损失
    
    结合两种策略:
    1. 在线挖掘: 从当前batch中动态选择最难的负样本
    2. 检索增强: 使用预检索的难负样本（如果提供）
    """
    def __init__(self, temp=0.07, hard_ratio=0.5, hard_weight=0.5):
        """
        Args:
            temp: 温度参数
            hard_ratio: 难负样本占比 (0-1)
            hard_weight: 难负样本损失的权重
        """
        super().__init__()
        self.temp = nn.Parameter(temp * torch.ones([]))
        self.hard_ratio = hard_ratio
        self.hard_weight = hard_weight
    
    def forward(self, fusion_feats, target_feats, hard_negative_feats=None):
        """
        Args:
            fusion_feats: (B, D) - 融合特征 (参考图像 + 文本)
            target_feats: (B, 32, D) - 目标图像特征
            hard_negative_feats: (B, K, 32, D) - 难负样本特征 (可选)
        
        Returns:
            loss: 总损失
            metrics: 统计信息字典
        """
        B = fusion_feats.size(0)
        
        # 计算与目标图像的相似度
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1),  # (B, 1, 1, D)
            target_feats.permute(0, 2, 1)             # (B, D, 32)
        ).squeeze()  # (B, B, 32)
        
        sim_max, _ = sim_t2q.max(-1)  # (B, B) - 取最大相似度
        sim_max = sim_max / self.temp
        
        # ===== 1. 标准对比损失 =====
        labels = torch.arange(B, device=sim_max.device)
        loss_standard = F.cross_entropy(sim_max, labels)
        
        # ===== 2. 在线难负样本挖掘 =====
        num_hard = max(1, int(B * self.hard_ratio))
        hard_loss_online = 0
        hard_neg_sims_list = []
        
        for i in range(B):
            # 找到最难的负样本（相似度最高的非正样本）
            neg_mask = torch.ones(B, dtype=torch.bool, device=sim_max.device)
            neg_mask[i] = False
            
            neg_sims = sim_max[i][neg_mask]
            
            if len(neg_sims) > 0:
                # 选择top-k最难的负样本
                k = min(num_hard, len(neg_sims))
                hard_neg_sims, hard_neg_indices = neg_sims.topk(k)
                hard_neg_sims_list.append(hard_neg_sims.mean().item())
                
                # 对难负样本计算损失
                pos_sim = sim_max[i, i]
                
                # 使用logsumexp技巧避免数值不稳定
                hard_loss_online += -torch.log(
                    torch.exp(pos_sim) / 
                    (torch.exp(pos_sim) + torch.exp(hard_neg_sims).sum())
                )
        
        hard_loss_online = hard_loss_online / B
        
        # ===== 3. 检索增强的难负样本损失（如果提供）=====
        hard_loss_retrieved = 0
        if hard_negative_feats is not None:
            # hard_negative_feats: (B, K, 32, D)
            K = hard_negative_feats.size(1)
            
            for i in range(B):
                pos_sim = sim_max[i, i]
                
                # 计算与检索到的难负样本的相似度
                hard_neg_sim = torch.matmul(
                    fusion_feats[i:i+1].unsqueeze(1),  # (1, 1, D)
                    hard_negative_feats[i].permute(1, 0)  # (D, K*32)
                ).squeeze()  # (K*32,)
                
                hard_neg_sim = hard_neg_sim.view(K, 32)
                hard_neg_sim_max, _ = hard_neg_sim.max(-1)  # (K,)
                hard_neg_sim_max = hard_neg_sim_max / self.temp
                
                # 计算损失
                hard_loss_retrieved += -torch.log(
                    torch.exp(pos_sim) / 
                    (torch.exp(pos_sim) + torch.exp(hard_neg_sim_max).sum())
                )
            
            hard_loss_retrieved = hard_loss_retrieved / B
        
        # ===== 4. 组合损失 =====
        if hard_negative_feats is not None:
            # 如果有检索的难负样本，两种都用
            total_loss = (
                loss_standard + 
                self.hard_weight * hard_loss_online + 
                self.hard_weight * hard_loss_retrieved
            )
        else:
            # 只用在线挖掘
            total_loss = loss_standard + self.hard_weight * hard_loss_online
        
        # ===== 5. 统计信息 =====
        metrics = {
            'loss_standard': loss_standard.item(),
            'loss_hard_online': hard_loss_online.item(),
            'loss_hard_retrieved': hard_loss_retrieved.item() if hard_negative_feats is not None else 0.0,
            'avg_hard_neg_sim': sum(hard_neg_sims_list) / len(hard_neg_sims_list) if hard_neg_sims_list else 0.0,
            'num_hard_negatives': num_hard,
        }
        
        return total_loss, metrics


class AdaptiveHardNegativeLoss(nn.Module):
    """
    自适应难负样本损失
    
    根据训练进度动态调整难负样本的权重和数量
    """
    def __init__(self, temp=0.07, initial_hard_ratio=0.3, max_hard_ratio=0.7, 
                 initial_hard_weight=0.3, max_hard_weight=0.7):
        super().__init__()
        self.temp = nn.Parameter(temp * torch.ones([]))
        self.initial_hard_ratio = initial_hard_ratio
        self.max_hard_ratio = max_hard_ratio
        self.initial_hard_weight = initial_hard_weight
        self.max_hard_weight = max_hard_weight
        self.current_epoch = 0
        self.total_epochs = 10  # 默认值，会在训练时更新
    
    def set_epoch(self, epoch, total_epochs):
        """更新当前epoch"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def get_adaptive_params(self):
        """根据训练进度计算自适应参数"""
        progress = self.current_epoch / max(1, self.total_epochs)
        
        # 线性增加难负样本的比例和权重
        hard_ratio = self.initial_hard_ratio + (self.max_hard_ratio - self.initial_hard_ratio) * progress
        hard_weight = self.initial_hard_weight + (self.max_hard_weight - self.initial_hard_weight) * progress
        
        return hard_ratio, hard_weight
    
    def forward(self, fusion_feats, target_feats, hard_negative_feats=None):
        """
        Args:
            fusion_feats: (B, D)
            target_feats: (B, 32, D)
            hard_negative_feats: (B, K, 32, D) - 可选
        """
        B = fusion_feats.size(0)
        hard_ratio, hard_weight = self.get_adaptive_params()
        
        # 计算相似度
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1),
            target_feats.permute(0, 2, 1)
        ).squeeze()
        sim_max, _ = sim_t2q.max(-1)
        sim_max = sim_max / self.temp
        
        # 标准损失
        labels = torch.arange(B, device=sim_max.device)
        loss_standard = F.cross_entropy(sim_max, labels)
        
        # 在线难负样本挖掘
        num_hard = max(1, int(B * hard_ratio))
        hard_loss = 0
        
        for i in range(B):
            neg_mask = torch.ones(B, dtype=torch.bool, device=sim_max.device)
            neg_mask[i] = False
            neg_sims = sim_max[i][neg_mask]
            
            if len(neg_sims) > 0:
                k = min(num_hard, len(neg_sims))
                hard_neg_sims = neg_sims.topk(k)[0]
                pos_sim = sim_max[i, i]
                
                hard_loss += -torch.log(
                    torch.exp(pos_sim) / 
                    (torch.exp(pos_sim) + torch.exp(hard_neg_sims).sum())
                )
        
        hard_loss = hard_loss / B
        
        # 组合损失
        total_loss = loss_standard + hard_weight * hard_loss
        
        metrics = {
            'loss_standard': loss_standard.item(),
            'loss_hard': hard_loss.item(),
            'hard_ratio': hard_ratio,
            'hard_weight': hard_weight,
            'epoch_progress': self.current_epoch / max(1, self.total_epochs),
        }
        
        return total_loss, metrics
