import torch
import torch.nn.functional as F

# 这种方式实现的Sinkhorn算法可以在GPU上运行
# 同时也可以进行梯度反传,且支持多batch操作
# 如不需要计算梯度,则可用with torch.no_grad()包围

def Sinkhorn(u, v, cost, reg=0.05, numItermax=100, stopThr=5e-5):
    K = torch.exp(-cost / reg)
    # torch.ones_like可以拷贝源张量的设备信息
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    for i in range(numItermax):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < stopThr:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T

def calc_similarity(anchor, anchor_center, fb, fb_center, stage, use_uniform=False):
    '''
    anchor: [C, R]
    anchor_center: [C]
    fb: [N, C, R]
    fb_center: [N, C]

    '''
    
    if stage == 0:
        sim = torch.einsum('c,nc->n', anchor_center, fb_center)
    else:
        N, _, R = fb.size()

        # 这里做的是cos相似性
        sim = torch.einsum('cm,ncs->nsm', anchor, fb).contiguous().view(N, R, R)
        dis = 1.0 - sim
        
        # 设置边缘分布为均匀分布
        if use_uniform:
            u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
            v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
        # 按论文的方式设置均匀分布
        else:
            att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
            u = att / (att.sum(dim=1, keepdims=True) + 1e-5)

            att = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
            v = att / (att.sum(dim=1, keepdims=True) + 1e-5)

        T = Sinkhorn(dis, u, v)
        sim = torch.sum(T * sim, dim=(1, 2))
    return sim

