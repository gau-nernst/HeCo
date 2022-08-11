import torch
import torch.nn.functional as F

from .contrast import Contrast
from .heco import HeCo


class ContrastDrop(Contrast):
    def forward(self, z_mp1, z_mp2, z_sc1, z_sc2, pos, beta1, beta2, dcl):
        z_proj_mp1 = self.proj(z_mp1)
        z_proj_sc1 = self.proj(z_sc1)
        matrix_mp1sc1 = self.sim(z_proj_mp1, z_proj_sc1)
        matrix_sc1mp1 = matrix_mp1sc1.t()

        z_proj_mp2 = self.proj(z_mp2)
        z_proj_sc2 = self.proj(z_sc2)

        matrix_sc1sc2 = self.sim(z_proj_sc1, z_proj_sc2)
        matrix_mp1mp2 = self.sim(z_proj_mp1, z_proj_mp2)

        # decoupled contrastive loss
        if dcl:
            # with multiple special positive pairs
            pos_sc1mp1_1 = torch.sum(matrix_sc1mp1.mul(pos.to_dense()), dim=1).view(-1, 1)
            deno_sc1mp1_1 = torch.sum(matrix_sc1mp1, dim=-1).view(-1, 1) - pos_sc1mp1_1 + 1e-8
            matrix_sc1mp1_1 = (matrix_sc1mp1) / deno_sc1mp1_1
            pos_sc1sc2_1 = torch.sum(matrix_sc1sc2.mul(pos.to_dense()), dim=1).view(-1, 1)
            deno_sc1sc2_1 = torch.sum(matrix_sc1sc2, dim=-1).view(-1, 1) - pos_sc1sc2_1 + 1e-8
            matrix_sc1sc2_1 = (matrix_sc1sc2) / deno_sc1sc2_1
            lori_sc = beta1 * (-torch.log(matrix_sc1mp1_1.mul(pos.to_dense()).sum(dim=-1)).mean()) + (1 - beta1) * (
                -torch.log(matrix_sc1sc2_1.mul(pos.to_dense()).sum(dim=-1)).mean())

            pos_mp1sc1_1 = torch.sum(matrix_mp1sc1.mul(pos.to_dense()), dim=1).view(-1, 1)
            deno_mp1sc1_1 = torch.sum(matrix_mp1sc1, dim=-1).view(-1, 1) - pos_mp1sc1_1 + 1e-8
            matrix_mp1sc1_1 = (matrix_mp1sc1) / deno_mp1sc1_1
            pos_mp1mp2_1 = torch.sum(matrix_mp1mp2.mul(pos.to_dense()), dim=1).view(-1, 1)
            deno_mp1mp2_1 = torch.sum(matrix_mp1mp2, dim=-1).view(-1, 1) - pos_mp1mp2_1 + 1e-8
            matrix_mp1mp2_1 = (matrix_mp1mp2) / deno_mp1mp2_1
            lori_mp = beta2 * (-torch.log(matrix_mp1sc1_1.mul(pos.to_dense()).sum(dim=-1)).mean()) + (1 - beta2) * (
                -torch.log(matrix_mp1mp2_1.mul(pos.to_dense()).sum(dim=-1)).mean())

            # with one special positive pairs
            # pos_sc1mp1_1 = torch.sum(matrix_sc1mp1.mul(pos.to_dense()), dim=1).view(-1, 1)
            # deno_sc1mp1_1 = torch.sum(matrix_sc1mp1, dim=-1).view(-1, 1) - pos_sc1mp1_1 + 1e-8
            # matrix_sc1mp1_1 = (matrix_sc1mp1) / deno_sc1mp1_1
            # deno_sc1sc2_1 = torch.sum(matrix_sc1sc2, dim=-1).view(-1, 1) - matrix_sc1sc2 + 1e-8
            # matrix_sc1sc2_1 = (matrix_sc1sc2) / deno_sc1sc2_1
            # lori_sc = beta1 * (-torch.log(matrix_sc1mp1_1.mul(pos.to_dense()).sum(dim=-1)).mean()) + (1 - beta1) * (
            #     -torch.log(matrix_sc1sc2_1.mul(pos.to_dense()).sum(dim=-1)).mean())
            #
            # pos_mp1sc1_1 = torch.sum(matrix_mp1sc1.mul(pos.to_dense()), dim=1).view(-1, 1)
            # deno_mp1sc1_1 = torch.sum(matrix_mp1sc1, dim=-1).view(-1, 1) - pos_mp1sc1_1 + 1e-8
            # matrix_mp1sc1_1 = (matrix_mp1sc1) / deno_mp1sc1_1
            # deno_mp1mp2_1 = torch.sum(matrix_mp1mp2, dim=-1).view(-1, 1) - matrix_mp1mp2 + 1e-8
            # matrix_mp1mp2_1 = (matrix_mp1mp2) / deno_mp1mp2_1
            # lori_mp = beta2 * (-torch.log(matrix_mp1sc1_1.mul(pos.to_dense()).sum(dim=-1)).mean()) + (1 - beta2) * (
            #     -torch.log(matrix_mp1mp2_1.mul(pos.to_dense()).sum(dim=-1)).mean())
        
        # without decoupled contrastive loss
        else:
            # # with only one special positive pair
            # matrix_sc1mp1_1 = (matrix_sc1mp1) / (torch.sum(matrix_sc1mp1, dim=1).view(-1, 1) + 1e-8)
            # matrix_sc1sc2_1 = (matrix_sc1sc2) / (torch.sum(matrix_sc1sc2, dim=1).view(-1, 1) + 1e-8)
            # lori_sc = beta1 * (-torch.log(matrix_sc1mp1_1.mul(pos.to_dense()).sum(dim=-1)).mean()) + (1 - beta1) * (
            #     -torch.log(matrix_sc1sc2_1).mean())
            #
            # matrix_mp1sc1_1 = (matrix_mp1sc1) / (torch.sum(matrix_mp1sc1, dim=1).view(-1, 1) + 1e-8)
            # matrix_mp1mp2_1 = (matrix_mp1mp2) / (torch.sum(matrix_mp1mp2, dim=1).view(-1, 1) + 1e-8)
            # lori_mp = beta2 * (-torch.log(matrix_mp1sc1_1.mul(pos.to_dense()).sum(dim=-1)).mean()) + (1 - beta2) * (
            #     -torch.log(matrix_mp1mp2_1).mean())

            # with multiple special positive pairs
            matrix_sc1mp1_1 = (matrix_sc1mp1) / (torch.sum(matrix_sc1mp1, dim=1).view(-1, 1) + 1e-8)
            matrix_sc1sc2_1 = (matrix_sc1sc2) / (torch.sum(matrix_sc1sc2, dim=1).view(-1, 1) + 1e-8)
            lori_mp = beta1 * (-torch.log(matrix_sc1mp1_1.mul(pos.to_dense()).sum(dim=-1)).mean()) + (1 - beta1) * (
                -torch.log(matrix_sc1sc2_1.mul(pos.to_dense()).sum(dim=-1)).mean())

            matrix_mp1sc1_1 = (matrix_mp1sc1) / (torch.sum(matrix_mp1sc1, dim=1).view(-1, 1) + 1e-8)
            matrix_mp1mp2_1 = (matrix_mp1mp2) / (torch.sum(matrix_mp1mp2, dim=1).view(-1, 1) + 1e-8)
            lori_sc = beta2 * (-torch.log(matrix_mp1sc1_1.mul(pos.to_dense()).sum(dim=-1)).mean()) + (1 - beta2) * (
                -torch.log(matrix_mp1mp2_1.mul(pos.to_dense()).sum(dim=-1)).mean())

        return self.lam * lori_mp + (1 - self.lam) * lori_sc


class HeCoDrop(HeCo):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam):
        super().__init__(hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam)
        self.contrast = ContrastDrop(hidden_dim, tau, lam)

    def forward(self, feats, pos, mps, nei_index, beta1, beta2, dcl):
        h_all = [F.elu(self.feat_drop(fc(feat))) for fc, feat in zip(self.fc_list, feats)]
        h1_all = [F.elu(self.feat_drop(fc(feat))) for fc, feat in zip(self.fc_list, feats)]
        z_mp1 = self.mp(h_all[0], mps)
        z_sc1 = self.sc(h_all, nei_index)
        z_mp2 = self.mp(h1_all[0], mps)
        z_sc2 = self.sc(h1_all, nei_index)
        loss = self.contrast(z_mp1, z_mp2, z_sc1, z_sc2, pos, beta1, beta2, dcl)
        return loss
