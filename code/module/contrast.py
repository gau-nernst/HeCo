import torch
import torch.nn as nn
import torch.nn.functional as F


def contrast(mat: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
    pos_values = mat[pos_idx[0], pos_idx[1]]
    pos_sum_exp = torch.zeros(mat.shape[0], device=mat.device)
    pos_sum_exp = pos_sum_exp.scatter_add_(0, pos_idx[0], pos_values.exp())
    loss = -pos_sum_exp.log() + mat.logsumexp(1)
    return loss.mean()


def decoupled_contrast(mat: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
    mat_exp = mat.exp()
    pos_values = mat_exp[pos_idx[0], pos_idx[1]]
    pos_sum_exp = torch.zeros(mat.shape[0], device=mat.device)
    pos_sum_exp = pos_sum_exp.scatter_add_(0, pos_idx[0], pos_values)
    neg_sum_exp = mat_exp.sum(1) - pos_sum_exp
    loss = -pos_sum_exp.log() + neg_sum_exp.log()
    return loss.mean()


class Contrast(nn.Module):
    def __init__(self, dcl: bool, tau: float, lam: float):
        super().__init__()
        self.contrast = decoupled_contrast if dcl else contrast
        self.tau_inv = 1 / tau
        self.lam = lam

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return self.tau_inv * z1 @ z2.t()

    def forward(self, z_mp: torch.Tensor, z_sc: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        matrix_mp2sc = self.sim(z_mp, z_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        
        loss_mp = self.contrast(matrix_mp2sc, pos)
        loss_sc = self.contrast(matrix_sc2mp, pos)
        return self.lam * loss_mp + (1 - self.lam) * loss_sc


class ContrastDrop(Contrast):
    def __init__(self, dcl: bool, tau: float, lam: float, beta1: float, beta2: float):
        super().__init__(dcl, tau, lam)
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(
        self,
        z_mp1: torch.Tensor,
        z_mp2: torch.Tensor,
        z_sc1: torch.Tensor, 
        z_sc2: torch.Tensor,
        pos: torch.Tensor
    ) -> torch.Tensor:
        matrix_mp1_sc1 = self.sim(z_mp1, z_sc1)
        matrix_sc1_mp1 = matrix_mp1_sc1.t()

        matrix_mp1_mp2 = self.sim(z_mp1, z_mp2)
        matrix_sc1_sc2 = self.sim(z_sc1, z_sc2)

        loss_sc1_mp1 = self.contrast(matrix_sc1_mp1, pos)
        loss_sc1_sc2 = self.contrast(matrix_sc1_sc2, pos)
        loss_sc = self.beta1 * loss_sc1_mp1 + (1 - self.beta1) * loss_sc1_sc2

        loss_mp1_sc1 = self.contrast(matrix_mp1_sc1, pos)
        loss_mp1_mp2 = self.contrast(matrix_mp1_mp2, pos)
        loss_mp = self.beta2 * loss_mp1_sc1 + (1 - self.beta2) * loss_mp1_mp2

        return self.lam * loss_mp + (1 - self.lam) * loss_sc
