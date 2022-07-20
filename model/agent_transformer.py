import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import cv2
import math
import numpy as np
from model.ops.modules import MSDeformAttn
from model.positional_encoding import SinePositionalEncoding
from model.sinkhorn import Sinkhorn

from model.transformer_mymha import MultiHeadAttentionOne

import torch.nn as nn

class OT_enhance_similarity(nn.Module):
    def __init__(self):
        super(OT_enhance_similarity,self).__init__()

    def forward(self, feat, prototypes, masks):
        '''
        feat:[b,num,c]
        prototypes: [b,k,c]
        masks:[b, mun]
        '''
        b, num, c = feat.shape
        _, k, _ = prototypes.shape
        sim = F.relu(self.get_similarity_OT(feat, prototypes, masks))
        return sim

    def get_cosine_similarity(self, feat, prototypes):
        '''
        feat:[b,num,c]
        prototypes: [b,k,c]
        '''
        feat_L2 = F.normalize(feat, p=2, dim=2)
        prototypes_L2 = F.normalize(prototypes, p=2, dim=2)
        att = torch.einsum('bnc,bkc->bnk',[feat_L2, prototypes_L2])

        return att


    def get_similarity_OT(self, feat, prototypes, masks):
        '''
        feat:[b,num,c]
        prototypes: [b,k,c]
        masks:[b, num]
        '''
        b, num, c = feat.shape
        _, k, _ = prototypes.shape
 
        att = self.get_cosine_similarity(feat, prototypes) #[b, num, k]
        num_bg = torch.sum((masks==0), dim=-1, keepdim=True) #[b, num]
        num_fg = num - num_bg

        trash = torch.zeros((b, num, 1)).cuda()
        trash[masks>0] = 2 
        cost_matrix = torch.cat([1 - att, trash], dim=2)
 
        col = torch.ones(b, num).cuda() / num
 
        row = torch.ones(b, k + 1).cuda()
        row[:, 0:k] = num_fg / k
        row[:, k:] = num_bg 
        row /= num
 
        T = Sinkhorn(col, row, cost_matrix, reg=0.05, numItermax=100, stopThr=1e-1) # [b, f*h*w, k+1]
        T *= num

        OT_eight = T[:, :, :-1]
        OT_att = OT_eight * att

        return OT_eight

class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

class MyCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert num_heads==1, "currently only implement num_heads==1"
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.s_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.seed_q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.seed_k_fc = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.s2p_drop   = nn.Dropout(attn_drop)
        self.p2q_drop   = nn.Dropout(attn_drop)
        self.align_drop = nn.Dropout(0.1)

        self.num_queries = 14
        self.num_cross_layers = 2
        self.num_self_layers = 1
        self.num_my_heads = 8
        self.drop_prob = 0.1



        # supp_self_layer
        self.transformer_cross = MultiHeadAttentionOne(n_head=num_heads, d_model=head_dim, d_k=512, d_v=512)
        self.ot_layer = OT_enhance_similarity()


    def forward(self, q, k, v, supp_valid_mask=None, supp_mask=None, supp_pos=None, query_pos=None, seed_vector=None, cyc=True): 
        B, N, C = q.shape
        query_seed = seed_vector.permute(0,2,1).contiguous()
 
        q = self.q_fc(q) # [b,nq,c]
        k = self.k_fc(k) # [b,ns,c]
        v = self.v_fc(v) # [b,ns,c]
        query_seed = self.s_fc(query_seed) # query_seed [b,n,c]
        query_seed = self.transformer_cross(query_seed, k.permute(0,2,1).contiguous(), v.permute(0,2,1).contiguous(), supp_mask) # b n c
        prototypes = query_seed.clone()
        prototypes_k = self.seed_k_fc(prototypes)
        prototypes_q = self.seed_q_fc(prototypes)

        att_q2p = self.ot_layer.get_cosine_similarity(q, prototypes_k) #[B,N,c],[B,k,C]->[B, Nq, Np]
        att_s2p = self.ot_layer(k, prototypes_q, supp_mask) #[b,Ns,Np]

        sim_id_s2p = att_s2p.max(2)[1] # [b, ns]
        sim_id_q2p = att_q2p.max(2)[1] # [b, nq]

        attn = torch.einsum('bnk,bmk->bnm',[att_q2p, att_s2p]) # [b, nq, ns]
        not_align = []
        align_matrix = torch.einsum('bs,bq->bsq', sim_id_s2p+1.5, sim_id_q2p+1.5) 
        for b_id in range(B):
            align_mask = torch.sqrt(align_matrix[b_id]) 
            mask = torch.ones_like(align_mask)
            mask[torch.where((2*align_mask) % 1 == 0)] = 0
            not_align.append(mask)

        not_align = torch.stack(not_align, dim=0).permute(0,2,1).contiguous() 
        not_align = self.align_drop(not_align)

        if supp_valid_mask is not None:
            supp_valid_mask = supp_valid_mask.unsqueeze(1).float()
            supp_valid_mask = supp_valid_mask * -10000.0 
            attn = attn + supp_valid_mask
            attn = attn + not_align * -10000.0
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class AgentTransformer(nn.Module):
    def __init__(self,
                 embed_dims=384, 
                 num_heads=1, 
                 num_layers=2,
                 num_levels=1,
                 num_points=9,
                 use_ffn=True,
                 dropout=0.1,
                 shot=1,
                 rand_fg_num=300, 
                 rand_bg_num=300, 
                 ):
        super(AgentTransformer, self).__init__()
        self.embed_dims             = embed_dims
        self.num_heads              = num_heads
        self.num_layers             = num_layers
        self.num_levels             = num_levels
        self.num_points             = num_points
        self.use_ffn                = use_ffn
        self.feedforward_channels   = embed_dims*3
        self.dropout                = dropout
        self.shot                   = shot
        self.use_cross              = True
        self.use_self               = True

        self.rand_fg_num = rand_fg_num * shot
        self.rand_bg_num = rand_bg_num * shot

        if self.use_cross:
            self.cross_layers = []
        self.qry_self_layers  = []
        self.supp_self_layers = [] #wy
        self.layer_norms = []
        self.ffns = []
        for l_id in range(self.num_layers):
            if self.use_cross:
                
                self.layer_norms.append(nn.LayerNorm(embed_dims)) # for supp self att
                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))
                    
                self.cross_layers.append(
                    MyCrossAttention(embed_dims, attn_drop=self.dropout, proj_drop=self.dropout),
                )
                
                self.layer_norms.append(nn.LayerNorm(embed_dims)) # for cross att
                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))
            
            if self.use_self:
                self.qry_self_layers.append(
                    MSDeformAttn(embed_dims, num_levels, num_heads, num_points)
                )
                self.supp_self_layers.append(
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=embed_dims, nhead=8),
                        num_layers=1
                    )
                )
                
                self.layer_norms.append(nn.LayerNorm(embed_dims)) # for query self att
                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))

        if self.use_cross: 
            self.cross_layers = nn.ModuleList(self.cross_layers)
        if self.use_self:
            self.qry_self_layers  = nn.ModuleList(self.qry_self_layers)
            self.supp_self_layers = nn.ModuleList(self.supp_self_layers)
        if self.use_ffn:
            self.ffns         = nn.ModuleList(self.ffns)
        self.layer_norms  = nn.ModuleList(self.layer_norms)

        self.positional_encoding = SinePositionalEncoding(embed_dims//2, normalize=True) 
        self.level_embed = nn.Parameter(torch.rand(num_levels, embed_dims)) #[1,384]
        self.supp_level_embed = nn.Parameter(torch.rand(num_levels, embed_dims)) #[1,384]
        nn.init.xavier_uniform_(self.level_embed)
        nn.init.xavier_uniform_(self.supp_level_embed)

        self.proj_drop  = nn.Dropout(dropout)
            

    def init_weights(self, distribution='uniform'):
        """Initialize the transformer weights."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes): # lvl仅有一个值

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_ 
            ref_x = ref_x.reshape(-1)[None] / W_  
            ref = torch.stack((ref_x, ref_y), -1) 
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1) # [1,num, 2]
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1) # level
        return reference_points

    def get_qry_flatten_input(self, x, qry_masks):
        src_flatten = [] 
        qry_valid_masks_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []        
        for lvl in range(self.num_levels):   
            src = x[lvl]
            bs, c, h, w = src.shape
            
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).permute(0, 2, 1) # [bs, c, h*w] -> [bs, h*w, c]
            src_flatten.append(src)

            if qry_masks is not None:
                qry_mask = qry_masks[lvl] #[b,h,w]
                qry_valid_mask = []
                qry_mask = F.interpolate(
                    qry_mask.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                for img_id in range(bs):
                    qry_valid_mask.append(qry_mask[img_id]==255) 
                qry_valid_mask = torch.stack(qry_valid_mask, dim=0) # [b,h,w]
            else:
                qry_valid_mask = torch.zeros((bs, h, w)) 

            pos_embed = self.positional_encoding(qry_valid_mask) 
            pos_embed = pos_embed.flatten(2).transpose(1, 2) 
            pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) 
            pos_embed_flatten.append(pos_embed)

            qry_valid_masks_flatten.append(qry_valid_mask.flatten(1))

        src_flatten = torch.cat(src_flatten, 1) # [bs, num_elem, c]
        qry_valid_masks_flatten = torch.cat(qry_valid_masks_flatten, dim=1) # [bs, num_elem]
        pos_embed_flatten = torch.cat(pos_embed_flatten, dim=1) # [bs, num_elem, c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # [num_lvl, 2]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [num_lvl]

        
        return src_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index

    def get_supp_flatten_input(self, s_x, supp_mask):
        s_x_flatten = []
        supp_valid_mask = []
        supp_obj_mask = []
        supp_mask = F.interpolate(supp_mask, size=s_x.shape[-2:], mode='nearest').squeeze(1) # [bs*shot, h, w]
        supp_mask = supp_mask.view(-1, self.shot, s_x.size(2), s_x.size(3))
        s_x = s_x.view(-1, self.shot, s_x.size(1), s_x.size(2), s_x.size(3)) #[b,st,c,h,w]
        supp_pos_embed_flatten = []


        for st_id in range(s_x.size(1)): #shot
            supp_valid_mask_s = []
            supp_obj_mask_s = []
            for img_id in range(s_x.size(0)):
                supp_valid_mask_s.append(supp_mask[img_id, st_id, ...]==255)
                obj_mask = supp_mask[img_id, st_id, ...]==1 
                if obj_mask.sum() == 0: 
                    obj_mask[obj_mask.size(0)//2-1:obj_mask.size(0)//2+1, obj_mask.size(1)//2-1:obj_mask.size(1)//2+1] = True
                if (obj_mask==False).sum() == 0: 
                    obj_mask[0, 0]   = False
                    obj_mask[-1, -1] = False 
                    obj_mask[0, -1]  = False
                    obj_mask[-1, 0]  = False
                supp_obj_mask_s.append(obj_mask)
            supp_valid_mask_s = torch.stack(supp_valid_mask_s, dim=0) # [bs, h, w]
            supp_pos_embed = self.positional_encoding(supp_valid_mask_s) #[b, h, w]
            supp_pos_embed = supp_pos_embed.flatten(2).transpose(1,2) #[b,hw,384]
            supp_pos_embed = supp_pos_embed + self.supp_level_embed[0].view(1,1,-1)
            supp_pos_embed_flatten.append(supp_pos_embed) 

            supp_valid_mask_s = supp_valid_mask_s.flatten(1) # [bs, h*w]
            supp_valid_mask.append(supp_valid_mask_s)

            supp_obj_mask_s = torch.stack(supp_obj_mask_s, dim=0)
            supp_obj_mask_s = (supp_obj_mask_s==1).flatten(1) # [bs, n]
            supp_obj_mask.append(supp_obj_mask_s)

            s_x_s = s_x[:, st_id, ...]
            s_x_s = s_x_s.flatten(2).permute(0, 2, 1)  # [bs, c, h*w] -> [bs, h*w, c]
            s_x_flatten.append(s_x_s)

        s_x_flatten = torch.cat(s_x_flatten, 1) # [bs, h*w*shot, c]
        supp_valid_mask = torch.cat(supp_valid_mask, 1)
        supp_mask_flatten = torch.cat(supp_obj_mask, 1)
        supp_pos_embed_flatten = torch.cat(supp_pos_embed_flatten, dim=1) #[b,s,h*w,c]

        return s_x_flatten, supp_valid_mask, supp_mask_flatten, supp_pos_embed_flatten

    def sparse_sampling(self, s_x, supp_mask, supp_valid_mask, supp_pos_embed_flatten,):
        # supp_pos_embed_flatten has the dim_size like s_x
        assert supp_mask is not None
        re_arrange_k = []
        re_arrange_mask = []
        re_arrange_valid_mask = []
        re_arrange_pos_embed = []
        for b_id in range(s_x.size(0)):
            k_b = s_x[b_id] # [num_elem, c]
            supp_mask_b = supp_mask[b_id] # [num_elem]
            num_fg = supp_mask_b.sum() 
            num_bg = (supp_mask_b==False).sum()
            fg_k = k_b[supp_mask_b] 
            bg_k = k_b[supp_mask_b==False] 

            if num_fg<self.rand_fg_num: 
                rest_num = self.rand_fg_num + self.rand_bg_num-num_fg 
                bg_select_idx = torch.randperm(num_bg)[:rest_num] 
                re_k = torch.cat([fg_k, bg_k[bg_select_idx]], dim=0) 
                re_mask = torch.cat([supp_mask_b[supp_mask_b==True], supp_mask_b[bg_select_idx]], dim=0) 
                re_valid_mask = torch.cat([supp_valid_mask[b_id][supp_mask_b==True], supp_valid_mask[b_id][bg_select_idx]], dim=0)
                re_pos_embed = torch.cat([supp_pos_embed_flatten[b_id][supp_mask_b==True], supp_pos_embed_flatten[b_id][bg_select_idx]], dim=0)

            elif num_bg<self.rand_bg_num:
                rest_num = self.rand_fg_num+self.rand_bg_num-num_bg
                fg_select_idx = torch.randperm(num_fg)[:rest_num]
                re_k = torch.cat([fg_k[fg_select_idx], bg_k], dim=0)
                re_mask = torch.cat([supp_mask_b[fg_select_idx], supp_mask_b[supp_mask_b==False]], dim=0)
                re_valid_mask = torch.cat([supp_valid_mask[b_id][fg_select_idx], supp_valid_mask[b_id][supp_mask_b==False]], dim=0)
                re_pos_embed = torch.cat([supp_pos_embed_flatten[b_id][fg_select_idx], supp_pos_embed_flatten[b_id][supp_mask_b==False]], dim=0)

            else:
                fg_select_idx = torch.randperm(num_fg)[:self.rand_fg_num]
                bg_select_idx = torch.randperm(num_bg)[:self.rand_bg_num]
                re_k = torch.cat([fg_k[fg_select_idx], bg_k[bg_select_idx]], dim=0)
                re_mask = torch.cat([supp_mask_b[fg_select_idx], supp_mask_b[bg_select_idx]], dim=0)
                re_valid_mask = torch.cat([supp_valid_mask[b_id][fg_select_idx], supp_valid_mask[b_id][bg_select_idx]], dim=0)
                re_pos_embed = torch.cat([supp_pos_embed_flatten[b_id][fg_select_idx], supp_pos_embed_flatten[b_id][bg_select_idx]], dim=0)

            re_arrange_k.append(re_k)
            re_arrange_mask.append(re_mask) 
            re_arrange_valid_mask.append(re_valid_mask)
            re_arrange_pos_embed.append(re_pos_embed)

        k = torch.stack(re_arrange_k, dim=0) #[b, n_fg+n_bg, c]
        supp_mask = torch.stack(re_arrange_mask, dim=0) #[bs, num_select]
        supp_valid_mask = torch.stack(re_arrange_valid_mask, dim=0) #[bs, num_select]
        supp_pos_embed = torch.stack(re_arrange_pos_embed, dim=0) #[b, nfg+ngb, c]

        return k, supp_mask, supp_valid_mask, supp_pos_embed

    def forward(self, x, qry_masks, s_x, supp_mask, seed_vector):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(qry_masks, list):
            qry_masks = [qry_masks.clone() for _ in range(self.num_levels)]

        assert len(x) == len(qry_masks) == self.num_levels
        bs, c = x[0].size()[:2]

        x_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index = self.get_qry_flatten_input(x, qry_masks)

        s_x, supp_valid_mask, supp_mask_flatten, supp_pos_embed_flatten = self.get_supp_flatten_input(s_x, supp_mask.clone()) 

        reference_points = self.get_reference_points(spatial_shapes, device=x_flatten.device) 

        q = x_flatten
        pos = pos_embed_flatten
        
        ln_id = 0
        ffn_id = 0
        for l_id in range(self.num_layers): #2
            if self.use_self: # self-att
                q =  q + self.proj_drop(self.qry_self_layers[l_id](q + pos, reference_points, q, spatial_shapes, level_start_index, qry_valid_masks_flatten))
                q = self.layer_norms[ln_id](q)
                ln_id += 1
       
                if self.use_ffn:
                    q = self.ffns[ffn_id](q)
                    ffn_id += 1
                    q = self.layer_norms[ln_id](q)
                    ln_id += 1
                

            if self.use_cross:
                k, sampled_mask, sampled_valid_mask, sampled_pos_embed = self.sparse_sampling(s_x, supp_mask_flatten, supp_valid_mask, supp_pos_embed_flatten) if self.training or l_id==0 else (k, sampled_mask, sampled_valid_mask, sampled_pos_embed)
                
                # supp self att
                self_out = k.permute(1,0,2).contiguous() # N B C
                self_out = self.supp_self_layers[l_id](self_out) # (src, mask, src_key_padding_mask)
                self_out = self_out.permute(1,0,2).contiguous() # B N C
                k = k + self_out
                k = self.layer_norms[ln_id](k)
                ln_id += 1
                
                if self.use_ffn:
                    k = self.ffns[ffn_id](k)
                    ffn_id += 1
                    k = self.layer_norms[ln_id](k)
                    ln_id += 1
                v = k.clone() #torch.Size([6, 600, 384]) # b n c
                cross_out = self.cross_layers[l_id](q, k, v, sampled_valid_mask, sampled_mask, sampled_pos_embed, pos, seed_vector)
                q = cross_out + q
                q = self.layer_norms[ln_id](q)
                ln_id += 1

                if self.use_ffn:
                    q = self.ffns[ffn_id](q)
                    ffn_id += 1
                    q = self.layer_norms[ln_id](q)
                    ln_id += 1

        qry_feat = q.permute(0, 2, 1) # [bs, c, num_ele]
        qry_feat_decouple = []
        for lvl in range(self.num_levels):
            start_idx = level_start_index[lvl].long()
            h, w = spatial_shapes[lvl]
            qry_feat_decouple.append(qry_feat[:, :, start_idx:start_idx+h*w].view(bs, c, h, w))

        return qry_feat_decouple


