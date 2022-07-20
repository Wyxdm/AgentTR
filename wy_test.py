import torch
from torch.nn import functional as F
B = 43
q = torch.tensor([[1,2,3,3,9,2,1]])
s = torch.tensor([[2,3,1,2]])

q = q + 0.5
s = s + 0.5

c = torch.mm(s.t(), q)
d = torch.einsum('bq,bs->bsq', q,s)
m = torch.sqrt(d)
x = torch.zeros_like(m)
print(c)
print(m)
x[torch.where((2*m % 1) == 0)] = 1
print(x)
# supp_mask = torch.tensor([[0,0,0,0,0,0,0],
#                           [0,0,1,1,1,0,0],
#                           [0,0,0,1,1,0,0],
#                           [0,0,1,1,1,0,0]]).bool() #【4，7】
# print("sup",supp_mask.repeat(2,1))
# supp_valid_mask = torch.tensor([[1,0,0,0,0,0,1],
#                           [1,0,0,0,0,1,1],
#                           [1,0,1,0,0,0,1],
#                           [0,0,0,0,0,0,1]]).bool()
# attn_mask_supp_ = torch.ones_like(supp_mask)  # [4,600]
# fg_attn_mask = torch.ones_like(supp_mask)
# fg_attn_mask[torch.where(supp_mask==True)] = False #fg is False, others True #[4,600]
# bg_attn_mask = torch.zeros_like(supp_mask)
# bg_attn_mask[torch.where(supp_mask==True)] = True
# bg_attn_mask[torch.where(supp_valid_mask==True)] = True # padding and fg pixels is True, bg is False  [4,600]
#
# bg_true = ~bg_attn_mask
# # valid_true = ~supp_valid_mask
# # print(bg_true)
#
# fg_attn_mask[torch.where(fg_attn_mask.sum(-1) == fg_attn_mask.shape[-1])] = False  # to_avoid_nan #[4,600]
# bg_attn_mask[torch.where(bg_attn_mask.sum(-1) == bg_attn_mask.shape[-1])] = False  # to_avoid_nan #[4,600]
# attn_mask_supp_ = attn_mask_supp_.unsqueeze(1).repeat(1,7,1)  # [4,7,600]
#
# # test = torch.zeros([7,7])
# # test[torch.where(supp_mask[1])] = 1
# # print(test)
#
# for bz in range(B):
#     attn_mask_supp_[bz][torch.where(supp_mask[bz])] = fg_attn_mask[bz]
#     attn_mask_supp_[bz][torch.where(bg_true[bz])] = bg_attn_mask[bz]
#     attn_mask_supp_[bz][torch.where(supp_valid_mask[bz])] = False
#     print(attn_mask_supp_[bz])