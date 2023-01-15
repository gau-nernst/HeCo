import math
from typing import Optional
from functools import partial

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
    def __init__(self, temp: float, margin: float, new: bool):
        assert margin < 0.5 * math.pi
        super().__init__()
        self.t_inv = 1.0 / temp
        self.margin = margin
        self.new = new
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin

    def forward(self, x1: Tensor, x2: Tensor, pos_mat: Optional[Tensor] = None, two_way: bool = True):
        x1_x2 = sim(x1, x2)
        new_x1_x2 = self.update_logits(x1_x2, pos_mat) * self.t_inv
        loss = InfoNCE.forward_logits(new_x1_x2, pos_mat)
        if two_way:
            if pos_mat is None:
                new_x2_x1 = new_x1_x2.t()
            else:
                # pos_mat may not be symmetric, so we can't just transpose new_x1_x2
                new_x2_x1 = self.update_logits(x1_x2.t(), pos_mat) * self.t_inv
            loss2 = InfoNCE.forward_logits(new_x2_x1, pos_mat)
            loss = (loss + loss2) * 0.5
        return loss

    def update_logits(self, logits: Tensor, pos_mat: Optional[Tensor]):
        # due to floating point precision, logits can be outside [-1, 1]
        # which will cause problems for arccos() and sin_theta calculation later
        logits = logits.clamp(min=-1, max=1)

        if pos_mat is not None:
            row_idx, col_idx = torch.nonzero(pos_mat, as_tuple=True)
        else:
            row_idx = col_idx = torch.arange(logits.shape[0], device=logits.device)
        cos_theta = logits[row_idx, col_idx]

        # https://github.com/deepinsight/insightface/commit/657ae30e41fc53641a50a68694009d0530d9f6b3
        if self.new:
            # new ArcFace
            new_logits = torch.cos(cos_theta.acos() + self.margin)

        else:
            # old ArcFace: there is a threshold
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
        n = x1.shape[0]
        x1_x2 = sim(x1, x2)

        if pos_mat is None:
            pos = x1_x2.diag().view(n, 1)
            mask = ~torch.eye(n, dtype=torch.bool)
            neg = x1_x2[mask].view(n, -1)
            loss = F.relu(pos - neg + self.margin).mean()

            if two_way:
                neg = x1_x2.t()[mask].view(n, -1)
                loss2 = F.relu(pos - neg + self.margin).mean()
                loss = (loss + loss2) * 0.5

        else:
            # since number of positives is not guaranteed to be the same for all nodes,
            # we have to loop over each node, which is very slow
            pos_mat = pos_mat.bool()

            loss = 0
            for i in range(n):
                pos = x1_x2[i, pos_mat[i]].view(-1, 1)
                neg = x1_x2[i, ~pos_mat[i]].view(1, -1)
                loss = loss + F.relu(pos - neg + self.margin).mean()
            loss = loss / n

            if two_way:
                x2_x1 = x1_x2.t()
                loss2 = 0
                for i in range(n):
                    pos = x2_x1[i, pos_mat[i]].view(-1, 1)
                    neg = x2_x1[i, ~pos_mat[i]].view(1, -1)
                    loss2 = loss2 + F.relu(pos - neg + self.margin).mean()
                loss2 = loss2 / n
                loss = (loss + loss2) * 0.5

        return loss


def build_loss(name: str, temp: float, margin: float):
    loss_mapping = dict(
        info_nce=InfoNCE,
        dcl=DCL,
        arcface=partial(ArcFace, new=True),
        arcface_old=partial(ArcFace, new=False),
        triplet=Triplet,
    )
    assert name in loss_mapping
    kwargs = dict()
    if name in ("info_nce", "dcl", "arcface", "arcface_old"):
        kwargs.update(temp=temp)
    if name in ("arcface", "arcface_old", "triplet"):
        kwargs.update(margin=margin)
    return loss_mapping[name](**kwargs)
