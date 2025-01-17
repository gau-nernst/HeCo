import argparse
import math
from typing import Optional, List
from functools import partial
import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_NEG_INF = -1e8


def sim(x1: Tensor, x2: Tensor) -> Tensor:
    return F.normalize(x1) @ F.normalize(x2).t()


class InfoNCE(nn.Module):
    def __init__(self, temp: float, soft_label: bool = False):
        super().__init__()
        self.t_inv = 1.0 / temp
        self.soft_label = soft_label

    def forward(self, x1: Tensor, x2: Tensor, pos_mat: Optional[Tensor] = None, two_way: bool = True):
        x1_x2 = sim(x1, x2) * self.t_inv
        loss = self.forward_logits(x1_x2, pos_mat, self.soft_label)
        if two_way:
            loss2 = self.forward_logits(x1_x2.t(), pos_mat, self.soft_label)
            loss = (loss + loss2) * 0.5
        return loss

    @staticmethod
    def forward_logits(logits: Tensor, pos_mat: Optional[Tensor], soft_label: bool):
        if pos_mat is None:
            labels = torch.arange(logits.size(0), device=logits.device)
            return F.cross_entropy(logits, labels)
        
        elif soft_label:
            return F.cross_entropy(logits, pos_mat / pos_mat.sum(1, keepdim=True))

        else:
            numerator_log = (logits + (1.0 - pos_mat) * _NEG_INF).logsumexp(1)
            denominator_log = logits.logsumexp(1)
            return (denominator_log - numerator_log).mean()


class DCL(InfoNCE):
    def __init__(self, temp: float, soft_label: bool = False):
        assert soft_label is False, "DCL does not support soft_label"
        super().__init__(temp, soft_label)
    
    @staticmethod
    def forward_logits(logits: Tensor, pos_mat: Optional[Tensor], soft_label: bool):
        if pos_mat is None:
            pos_mat = torch.eye(logits.size(0), device=logits.device)
            numerator_log = logits.diag()
        
        else:
            numerator_log = (logits + (1.0 - pos_mat) * _NEG_INF).logsumexp(1)

        denominator_log = (logits + pos_mat * _NEG_INF).logsumexp(1)
        return (denominator_log - numerator_log).mean()


# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
class ArcFace(nn.Module):
    def __init__(self, temp: float, margin: float, old: bool, soft_label: bool = False):
        assert margin < 0.5 * math.pi
        super().__init__()
        self.t_inv = 1.0 / temp
        self.soft_label = soft_label
        self.margin = margin
        self.old = old
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin

    def forward(self, x1: Tensor, x2: Tensor, pos_mat: Optional[Tensor] = None, two_way: bool = True):
        x1_x2 = sim(x1, x2)
        new_x1_x2 = self.update_logits(x1_x2, pos_mat) * self.t_inv
        loss = InfoNCE.forward_logits(new_x1_x2, pos_mat, self.soft_label)
        if two_way:
            if pos_mat is None:
                new_x2_x1 = new_x1_x2.t()
            else:
                # pos_mat may not be symmetric, so we can't just transpose new_x1_x2
                new_x2_x1 = self.update_logits(x1_x2.t(), pos_mat) * self.t_inv
            loss2 = InfoNCE.forward_logits(new_x2_x1, pos_mat, self.soft_label)
            loss = (loss + loss2) * 0.5
        return loss

    def update_logits(self, logits: Tensor, pos_mat: Optional[Tensor]):
        # due to floating point precision, logits can be outside [-1, 1]
        # which will cause problems for arccos() and sin_theta calculation later
        logits = logits.clamp(min=-1, max=1)

        if pos_mat is not None:
            row_idx, col_idx = torch.nonzero(pos_mat, as_tuple=True)
        else:
            row_idx = torch.arange(logits.shape[0], device=logits.device)
            col_idx = torch.arange(logits.shape[0], device=logits.device)
        cos_theta = logits[row_idx, col_idx]

        # https://github.com/deepinsight/insightface/commit/657ae30e41fc53641a50a68694009d0530d9f6b3
        if self.old:
            # old ArcFace: there is a threshold
            sin_theta = (1.0 - cos_theta ** 2) ** 0.5    
            cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

            # if cos(theta) > cos(pi - margin), use cos(theta + m), else use cos(theta) - sin(pi - margin) * margin
            # https://github.com/deepinsight/insightface/issues/2126
            new_logits = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.sinmm)

        else:
            # new ArcFace
            new_logits = torch.cos(cos_theta.acos() + self.margin)

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


# similar to Barlow Twins
class Regression(nn.Module):
    def __init__(self, lambd: float):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, x1: Tensor, x2: Tensor, pos_mat: Tensor = None, two_way: bool = True):
        n = x1.shape[0]
        if pos_mat is None:
            pos_mat = torch.eye(x1.shape[0], device=x1.device, dtype=torch.bool)
        else:
            pos_mat = pos_mat.bool()
        
        x1_x2 = sim(x1, x2)
        num_pos = pos_mat.sum(1)

        pos_loss = x1_x2[pos_mat].add(-1).square().sum(1).div(num_pos).mean()
        neg_loss = x1_x2[~pos_mat].square().sum(1).div(n - num_pos).mean()
        loss = pos_loss + neg_loss * self.lambd

        if two_way:
            x2_x1 = x1_x2.t()
            pos_loss = x2_x1[pos_mat].add(-1).square().sum(1).div(num_pos).mean()
            neg_loss = x2_x1[~pos_mat].square().sum(1).div(n - num_pos).mean()
            loss2 = pos_loss + neg_loss * self.lambd
            loss = (loss + loss2) * 0.5
        
        return loss


# https://github.com/facebookresearch/barlowtwins/blob/main/main.py
class BarlowTwins(nn.Module):
    def __init__(self, lambd: float):
        super().__init__()
        self.lambd = lambd

    def forward(self, x1: Tensor, x2: Tensor, pos_mat = None, two_way = True):
        # if pos_mat is not None:
        #     raise NotImplementedError("Multi positive is not supported for Barlow Twins")

        num_nodes, z_dim = x1.shape
        cross_corr = self.bn(x1).t() @ self.bn(x2) / num_nodes
        on_diag = cross_corr.diag().add(-1).square().sum()  # invariance term

        mask = ~torch.eye(z_dim, device=x1.device, dtype=torch.bool)
        off_diag = cross_corr[mask].square().sum()  # redundancy reduction term

        loss = on_diag + off_diag * self.lambd
        return loss

    @staticmethod
    def bn(x: Tensor):
        return F.batch_norm(x, None, None, training=True)


# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
class VICReg(nn.Module):
    def __init__(self, sim_coef: float, std_coef: float, cov_coef: float):
        super().__init__()
        self.sim_coef = sim_coef
        self.std_coef = std_coef
        self.cov_coef = cov_coef
    
    def forward(self, x1: Tensor, x2: Tensor, pos_mat = None, two_way = True):
        # if pos_mat is not None:
        #     raise NotImplementedError("Multi positive is not supported for Barlow Twins")
        
        sim_loss = F.mse_loss(x1, x2)  # this is before mean-centered

        x1 = x1 - x1.mean(0, keepdim=True)
        x2 = x2 - x2.mean(0, keepdim=True)

        std_loss = (self.std_loss(x1) + self.std_loss(x2)) * 0.5
        cov_loss = (self.cov_loss(x1) + self.cov_loss(x2)) * 0.5

        loss = sim_loss * self.sim_coef + std_loss * self.std_coef + cov_loss * self.cov_coef
        return loss

    @staticmethod
    def std_loss(x: Tensor):
        std = x.var(0).add(1e-4).sqrt()
        return F.relu(1 - std).mean()

    @staticmethod
    def cov_loss(x: Tensor):
        num_nodes, z_dim = x.shape
        cov = x.t() @ x / (num_nodes - 1)
        mask = ~torch.eye(z_dim, device=x.device, dtype=torch.bool)
        return cov[mask].square().sum() / z_dim


class DeepCluster(nn.Module):
    def __init__(self, temp: float, n_clusters: int, emb_dim: int):
        super().__init__()
        self.t_inv = 1.0 / temp
        self.counter = 0
        self.centroids = nn.Parameter(torch.empty(n_clusters, emb_dim))
        self.assignments = None

    def forward(self, x1: Tensor, x2: Tensor, pos_mat = None, two_way = True):
        logits = sim(x1, self.centroids) * self.t_inv
        return F.cross_entropy(logits, self.assignments)


class SpectralClustering(nn.Module):
    def __init__(self, temp, n_labels, emb_dim):
        super().__init__()
        self.t_inv = 1 / temp
        # self.fc = nn.Linear(emb_dim, n_labels, bias=False)
        self.prototypes = nn.Parameter(torch.empty(n_labels, emb_dim))
        nn.init.normal_(self.prototypes, 0, 1 / emb_dim ** 0.5)
        self.labels = None

    def forward(self, x1, x2, pos_mat = None, two_way = True):
        # logits = self.fc(x1)
        logits = sim(x1, self.prototypes) * self.t_inv
        return F.cross_entropy(logits, self.labels)


class CompositeLoss(nn.Module):
    def __init__(self, losses: List[nn.Module], loss_weights: List[float]):
        assert len(losses) == len(loss_weights)
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.loss_weights = loss_weights
    
    def forward(self, x1: Tensor, x2: Tensor, pos_mat = None, two_way = True):
        out = 0
        for loss, weight in zip(self.losses, self.loss_weights):
            out += loss(x1, x2, pos_mat, two_way) * weight
        return out


def build_loss(args: argparse.Namespace):
    loss_dict = dict(
        info_nce=partial(InfoNCE, args.temp, args.soft_label),
        dcl=partial(DCL, args.temp, args.soft_label),
        arcface=partial(ArcFace, args.temp, args.margin, False, args.soft_label),
        arcface_old=partial(ArcFace, args.temp, args.margin, True, args.soft_label),
        triplet=partial(Triplet, args.margin),
        regression=partial(Regression, args.lambd),
        barlow_twins=partial(BarlowTwins, args.lambd),
        vicreg=partial(VICReg, args.sim_coef, args.std_coef, args.cov_coef),
        deepcluster=partial(DeepCluster, args.temp, args.n_clusters, args.hidden_dim),
        spectral_clustering=partial(SpectralClustering, args.temp, args.n_clusters, args.hidden_dim),
    )
    if args.loss_type in loss_dict:
        return loss_dict[args.loss_type]()
    else:
        # e.g. "info_nce_0.5_barlow_twins_0.5"
        losses, loss_weights = zip(*re.findall(r"([a-z]\w+[a-z])_([\d\.]+)", args.loss_type))
        print(f"Composite loss: {losses} {loss_weights}")
        losses = [loss_dict[loss]() for loss in losses]
        loss_weights = [float(weight) for weight in loss_weights]
        return CompositeLoss(losses, loss_weights)
