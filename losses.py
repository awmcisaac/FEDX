"""
Functions to compute loss objectives of FedX.

"""

import torch

from utils import F

def nt_xent(x1, x2, t=0.1, device="cpu"):
    """Contrastive loss objective function
    Normalized temperature-scaled cross entropy"""
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    batch_size = x1.size(0)
    out = torch.cat([x1, x2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

def off_diagonal(M):
    res = M.clone()
    res.diagonal(dim1=-2, dim2=-1).zero_()
    return res

def bt_loss(x1, x2, l=1, device="cpu"):
    """Barlow Twins loss objective function"""
    batch_size = x1.size(0)
    emb_dim = x1.size(1)

    bn = torch.nn.BatchNorm1d(emb_dim, affine=False).to(device)
    x1 = bn(x1)
    x2 = bn(x2)

    c = torch.mm(x1.T, x2) / batch_size
    c_diff = torch.pow((c - torch.eye(emb_dim, device=device)), 2)
    c_diff[~torch.eye(emb_dim, device=device).bool()] *= l
    loss = c_diff.sum()
    return loss

def ss_loss(x1, x2, device="cpu"):
    """SimSiam loss objective function"""
    return NotImplementedError

def js_loss(x1, x2, xa, t=0.1, t2=0.01, device="cpu"):
    """Relational loss objective function
    Jensen-Shannon divergence"""
    pred_sim1 = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())
    inputs1 = F.log_softmax(pred_sim1 / t, dim=1)
    pred_sim2 = torch.mm(F.normalize(x2, dim=1), F.normalize(xa, dim=1).t())
    inputs2 = F.log_softmax(pred_sim2 / t, dim=1)
    target_js = (F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)) / 2
    js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
    js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
    return (js_loss1 + js_loss2) / 2.0
