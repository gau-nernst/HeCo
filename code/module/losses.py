import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_NEG_INF = -1e8


def sim(x1: Tensor, x2: Tensor) -> Tensor:
    return F.normalize(x1) @ F.normalize(x2).t()


class InfoNCE(nn.Module):
    def __init__(self, temp: float):
        super().__init__()
        self.t_inv = 1.0 / temp

    def forward(self, x1: Tensor, x2: Tensor, pos_mat: Optional[Tensor] = None, two_way: bool = True):
        x1_x2 = sim(x1, x2) * self.t_inv
        loss = self.forward_logits(x1_x2, pos_mat)
        if two_way:
            loss2 = self.forward_logits(x1_x2.t(), pos_mat)
            loss = (loss + loss2) * 0.5
        return loss

    @staticmethod
    def forward_logits(logits: Tensor, pos_mat: Optional[Tensor]):
        if pos_mat is None:
            labels = torch.arange(logits.size(0), device=logits.device)
            return F.cross_entropy(logits, labels)
        
        else:
            numerator_log = (logits + (1.0 - pos_mat) * _NEG_INF).logsumexp(1)
            denominator_log = logits.logsumexp(1)
            return (denominator_log - numerator_log).mean()


class DCL(InfoNCE):
    @staticmethod
    def forward_logits(logits: Tensor, pos_mat: Optional[Tensor]):
        if pos_mat is None:
            pos_mat = torch.eye(logits.size(0), device=logits.device)
            numerator_log = logits.diag()
        
        else:
            numerator_log = (logits + (1.0 - pos_mat) * _NEG_INF).logsumexp(1)

        denominator_log = (logits + pos_mat * _NEG_INF).logsumexp(1)
        return (denominator_log - numerator_log).mean()


# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
class ArcFace(nn.Module):
    def __init__(self, temp: float, margin: float):
        assert margin < 0.5 * math.pi
        super().__init__()
        self.t_inv = 1.0 / temp
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin

    def forward(self, x1: Tensor, x2: Tensor, pos_mat: Optional[Tensor] = None, two_way: bool = True):
        x1_x2 = sim(x1, x2)
        new_x1_x2 = self.update_logits(x1_x2, pos_mat) * self.t_inv
        loss = InfoNCE.forward_logits(new_x1_x2, pos_mat)
        if two_way:
            new_x2_x1 = new_x1_x2.t() if pos_mat is None else self.update_logits(x1_x2.t(), pos_mat) * self.t_inv
            loss2 = InfoNCE.forward_logits(new_x2_x1, pos_mat)
            loss = (loss + loss2) * 0.5
        return loss

    def update_logits(self, logits: Tensor, pos_mat: Optional[Tensor]):
        if pos_mat is None:
            pos_mat = torch.eye(logits.size(0), device=logits.device)
        row_idx, col_idx = torch.nonzero(pos_mat, as_tuple=True)
        cos_theta = logits[row_idx, col_idx]
        sin_theta = (1.0 - cos_theta ** 2) ** 0.5    
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # if cos(theta) > cos(pi - margin), use cos(theta + m), else use cos(theta) - sin(pi - margin) * margin
        # https://github.com/deepinsight/insightface/issues/2126
        new_logits = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.sinmm)
        logits[row_idx, col_idx] = new_logits
        return logits


class Triplet(nn.Module):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin
    
    def forward(self, x1: Tensor, x2: Tensor, pos_mat: Optional[Tensor] = None, two_way: bool = True):
        x1_x2 = sim(x1, x2)
        pos = x1_x2.diag().view(-1, 1)
        loss = F.relu(pos - x1_x2 + self.margin).sum(1).div(x1.shape[0] - 1).mean()
        if two_way:
            x2_x1 = x1_x2.t()
            loss2 = F.relu(pos - x2_x1 + self.margin).sum(1).div(x1.shape[0] - 1).mean()
            loss = (loss + loss2) * 0.5
        return loss


def build_loss(name: str, temp: float, margin: float):
    loss_mapping = dict(
        info_nce=InfoNCE,
        dcl=DCL,
        arcface=ArcFace,
        triplet=Triplet,
    )
    assert name in loss_mapping
    kwargs = dict()
    if name in ("info_nce", "dcl", "arcface"):
        kwargs.update(temp=temp)
    if name in ("arcface", "triplet"):
        kwargs.update(margin=margin)
    return loss_mapping[name](**kwargs)
