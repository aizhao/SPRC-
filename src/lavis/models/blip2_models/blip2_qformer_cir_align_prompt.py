"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from torchvision.ops import roi_align

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


class GatedFusionModule(nn.Module):
    """
    门控融合模块：让文本选择性地修改图像特征
    
    核心思想：
    1. 文本生成门控信号，决定哪些图像特征需要修改
    2. 文本生成修改向量，指导如何修改
    3. 通过门控机制平滑地融合原始特征和修改后的特征
    
    公式：
    gate = sigmoid(W_g * [image_feat; text_feat_projected])
    delta = tanh(W_d * text_feat_projected)
    fused_feat = image_feat + alpha * gate * delta
    
    注意：支持image_feat和text_feat维度不同的情况
    """
    def __init__(self, image_dim, text_dim, dropout=0.1):
        super().__init__()
        self.image_dim = image_dim  # 图像特征维度 (e.g., 1088)
        self.text_dim = text_dim    # 文本特征维度 (e.g., 768)
        
        # 文本投影：将文本特征投影到图像特征空间
        self.text_proj = nn.Linear(text_dim, image_dim)
        
        # 门控网络：决定修改的程度
        self.gate_net = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(image_dim, image_dim),
            nn.Sigmoid()  # 输出[0,1]的门控信号
        )
        
        # 修改网络：生成修改向量
        self.delta_net = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(image_dim, image_dim),
            nn.Tanh()  # 输出[-1,1]的修改向量
        )
        
        # 残差连接的权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, image_feat, text_feat):
        """
        Args:
            image_feat: (B, N, D_img) - 图像特征，D_img = image_dim
            text_feat: (B, M, D_txt) - 文本特征，D_txt = text_dim
        
        Returns:
            fused_feat: (B, N, D_img) - 融合后的特征
        """
        batch_size, num_img_tokens, _ = image_feat.shape
        _, num_txt_tokens, _ = text_feat.shape
        
        # 使用文本的平均池化作为全局文本表示
        text_global = text_feat.mean(dim=1, keepdim=True)  # (B, 1, D_txt)
        
        # 将文本投影到图像特征空间
        text_global_proj = self.text_proj(text_global)  # (B, 1, D_img)
        text_global_proj = text_global_proj.expand(-1, num_img_tokens, -1)  # (B, N, D_img)
        
        # 拼接图像和投影后的文本特征用于门控
        concat_feat = torch.cat([image_feat, text_global_proj], dim=-1)  # (B, N, 2*D_img)
        
        # 计算门控信号：决定每个位置修改的程度
        gate = self.gate_net(concat_feat)  # (B, N, D_img)
        
        # 计算修改向量：文本指导的修改方向
        delta = self.delta_net(text_global_proj)  # (B, N, D_img)
        
        # 门控融合：只在需要的地方应用修改
        fused_feat = image_feat + self.alpha * gate * delta
        
        return fused_feat


@registry.register_model("blip2_cir_align_prompt")
class Blip2QformerCirAlignPrompt(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        # new tokens
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        
        # RoI Align region feature projection
        # Use visual encoder's feature dimension instead of Q-former's hidden size
        self.region_proj = nn.Linear(self.visual_encoder.num_features, embed_dim)
        self.use_region_loss = False  # 可以通过配置文件控制
        
        # 门控融合模块 - 改进文本-图像融合机制
        # 注意：image_embeds来自visual_encoder，维度是visual_encoder.num_features (1088)
        #      text_embeds来自Qformer.bert.embeddings，维度是Qformer.config.hidden_size (768)
        self.gated_fusion = GatedFusionModule(
            image_dim=self.visual_encoder.num_features,  # 1088
            text_dim=self.Qformer.config.hidden_size,     # 768
            dropout=0.1
        )
        self.use_gated_fusion = True  # 是否使用门控融合
        
        # 难负样本挖掘参数
        self.use_hard_negatives = False  # 是否使用难负样本
        self.hard_negative_ratio = 0.5   # 难负样本占比
        self.hard_negative_weight = 0.5  # 难负样本损失权重


    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]

        ###============== reference text fusion ===================###
        # reference image feature  
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        # 门控融合：让文本选择性地修改图像特征
        if self.use_gated_fusion:
            # 先获取文本的初步表示
            text_embeds = self.Qformer.bert.embeddings(
                input_ids=text_tokens.input_ids
            )  # (B, L_text, D)
            
            # 应用门控融合
            image_embeds_fused = self.gated_fusion(image_embeds, text_embeds)
        else:
            image_embeds_fused = image_embeds
        
        # fusion reference image and text tokens into a set of multi-modal tokens
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds_fused,  # 使用融合后的图像特征
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )

        ###============== Fusion-target Contrastive ===================###
        # reference image feature  
        taregt_embeds = self.ln_vision(self.visual_encoder(target))
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )

        # 计算loss_itc
        # 支持难负样本：如果提供了hard_negative_targets，使用难负样本损失
        if 'hard_negative_targets' in samples and samples['hard_negative_targets'] is not None:
            # 使用难负样本增强的损失
            loss_itc = self._compute_hard_negative_loss(
                fusion_feats, 
                target_feats,
                samples.get('hard_negative_feats', None)
            )
        else:
            # 标准对比损失
            sim_t2q = torch.matmul(
                fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
            ).squeeze()

            sim_i2t, _ = sim_t2q.max(-1)
            sim_i2t = sim_i2t / self.temp
            bs = image.size(0)
            targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(
                image.device
            )
            loss_itc = F.cross_entropy(sim_i2t, targets)

         ###============== Relative Contrastive ===================###
        prompt_tokens = self.prompt_tokens.expand(image_embeds.shape[0], -1, -1)

        text_only_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=prompt_tokens,
            attention_mask=attention_mask,
            return_dict=True,
            no_img=True
        )

        text_only_feat = F.normalize(
            self.text_proj(text_only_output.last_hidden_state[:, 0, :]), dim=-1
        )

        sim_r2t = torch.matmul(
            text_only_feat.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_r2t, _ = sim_r2t.max(-1)
        sim_r2t = sim_r2t / self.temp
        loss_rtc = F.cross_entropy(sim_r2t, targets)

        loss_align = F.mse_loss(fusion_output.last_hidden_state[:, : query_tokens.size(1), :].mean(1), 
                                prompt_tokens.clone().detach().mean(1))

        losses = {
            'loss_itc': loss_itc, 
            'loss_rtc': loss_rtc,
            'loss_align': loss_align
        }
        
        # 如果提供了region boxes，计算区域级损失
        if self.use_region_loss and 'region_boxes' in samples and samples['region_boxes'] is not None:
            loss_region = self.compute_region_loss(
                image_embeds, taregt_embeds, 
                samples['region_boxes'], samples.get('target_region_boxes')
            )
            losses['loss_region'] = loss_region
        
        return losses
    
    def _compute_hard_negative_loss(self, fusion_feats, target_feats, hard_negative_feats=None):
        """
        计算难负样本增强的对比损失
        
        Args:
            fusion_feats: (B, D) - 融合特征
            target_feats: (B, 32, D) - 目标图像特征
            hard_negative_feats: (B, K, 32, D) - 难负样本特征（可选）
        
        Returns:
            loss: 难负样本增强的损失
        """
        B = fusion_feats.size(0)
        
        # 计算与目标的相似度
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1),
            target_feats.permute(0, 2, 1)
        ).squeeze()
        sim_max, _ = sim_t2q.max(-1)  # (B, B)
        sim_max = sim_max / self.temp
        
        # 标准对比损失
        labels = torch.arange(B, device=sim_max.device)
        loss_standard = F.cross_entropy(sim_max, labels)
        
        # 在线难负样本挖掘
        num_hard = max(1, int(B * self.hard_negative_ratio))
        hard_loss = 0
        
        for i in range(B):
            # 找到最难的负样本
            neg_mask = torch.ones(B, dtype=torch.bool, device=sim_max.device)
            neg_mask[i] = False
            neg_sims = sim_max[i][neg_mask]
            
            if len(neg_sims) > 0:
                k = min(num_hard, len(neg_sims))
                hard_neg_sims = neg_sims.topk(k)[0]
                pos_sim = sim_max[i, i]
                
                # 难负样本损失
                hard_loss += -torch.log(
                    torch.exp(pos_sim) / 
                    (torch.exp(pos_sim) + torch.exp(hard_neg_sims).sum())
                )
        
        hard_loss = hard_loss / B
        
        # 组合损失
        total_loss = loss_standard + self.hard_negative_weight * hard_loss
        
        return total_loss

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def extract_region_features(self, image_embeds, boxes, image_size=(224, 224)):
        """
        使用RoI Align从图像特征中提取区域特征
        
        Args:
            image_embeds: 图像特征 (B, N, D) 其中N是patch数量
            boxes: bounding boxes列表，每个元素是该图像的boxes (x1, y1, x2, y2)格式，归一化到[0,1]
            image_size: 原始图像大小
        
        Returns:
            region_features: 区域特征列表
        """
        batch_size = image_embeds.shape[0]
        hidden_dim = image_embeds.shape[-1]
        num_patches = image_embeds.shape[1]
        
        # 假设image_embeds是从ViT出来的patch特征，需要reshape成feature map
        # 对于224x224图像，patch_size=16，feature map是14x14
        # 注意：EVA-CLIP可能包含CLS token，需要去除
        feature_map_size = int((num_patches) ** 0.5)
        
        # 如果不是完全平方数，说明有CLS token，去除第一个token
        if feature_map_size * feature_map_size != num_patches:
            feature_map_size = int((num_patches - 1) ** 0.5)
            image_embeds_no_cls = image_embeds[:, 1:, :]  # 去除CLS token
        else:
            image_embeds_no_cls = image_embeds
        
        all_region_features = []
        
        for i in range(batch_size):
            if boxes[i] is None or len(boxes[i]) == 0:
                # 如果没有box，返回空tensor
                all_region_features.append(torch.empty(0, hidden_dim, device=image_embeds.device))
                continue
            
            # Reshape feature map: (N, D) -> (1, D, H, W)
            feat_map = image_embeds_no_cls[i].view(feature_map_size, feature_map_size, hidden_dim)
            feat_map = feat_map.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
            
            # 准备RoI boxes: 格式为 (batch_idx, x1, y1, x2, y2)
            # boxes已经是归一化的[0,1]，需要转换到feature map坐标
            rois = []
            for box in boxes[i]:
                x1, y1, x2, y2 = box
                # 转换到feature map坐标
                fx1 = x1 * feature_map_size
                fy1 = y1 * feature_map_size
                fx2 = x2 * feature_map_size
                fy2 = y2 * feature_map_size
                rois.append([0, fx1, fy1, fx2, fy2])  # batch_idx=0因为是单个图像
            
            rois_tensor = torch.tensor(rois, dtype=torch.float32, device=feat_map.device)
            
            # RoI Align提取区域特征
            pooled = roi_align(
                input=feat_map,
                boxes=rois_tensor,
                output_size=(1, 1),  # 池化到1x1
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True,
            )  # (num_boxes, D, 1, 1)
            
            region_feats = pooled.squeeze(-1).squeeze(-1)  # (num_boxes, D)
            all_region_features.append(region_feats)
        
        return all_region_features
    
    def compute_region_loss(self, ref_image_embeds, target_image_embeds, ref_boxes, target_boxes):
        """
        区域级对比损失 - 使用InfoNCE风格的对比学习
        
        策略：在batch内进行对比学习，让正确的(ref, target)对相似度高，
        而与batch内其他样本的相似度低
        
        Args:
            ref_image_embeds: 参考图像特征 (B, N, D)
            target_image_embeds: 目标图像特征 (B, N, D)
            ref_boxes: 参考图像的bounding boxes
            target_boxes: 目标图像的bounding boxes
        
        Returns:
            loss_region: 区域级对比损失
        """
        batch_size = ref_image_embeds.shape[0]
        
        # 使用CLS token作为全局特征
        ref_global = ref_image_embeds[:, 0, :]  # (B, D)
        tgt_global = target_image_embeds[:, 0, :]  # (B, D)
        
        # 投影并归一化
        ref_proj = F.normalize(self.region_proj(ref_global), dim=-1)  # (B, embed_dim)
        tgt_proj = F.normalize(self.region_proj(tgt_global), dim=-1)  # (B, embed_dim)
        
        # 计算相似度矩阵 (B, B)
        # sim_matrix[i, j] 表示第i个ref和第j个target的相似度
        sim_matrix = torch.matmul(ref_proj, tgt_proj.t()) / self.temp  # (B, B)
        
        # 对角线是正样本对，其他是负样本
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # 使用交叉熵损失（InfoNCE）
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit
    

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text):
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )


        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_i2t, _ = sim_t2q.max(-1)
        # sim_i2t, _ = torch.topk(sim_t2q, k=5, dim=-1)
        # sim_i2t = sim_i2t.mean(-1)
        return sim_i2t


    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
