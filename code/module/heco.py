import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrast import ContrastDrop


class HeCo(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        feats_dim_list: list[int],
        feat_drop: float,
        mp_enc: nn.Module,
        sc_enc: nn.Module,
        contrast: nn.Module,
    ):
        super(HeCo, self).__init__()
        self.fc_list = nn.ModuleList()
        for feats_dim in feats_dim_list:
            fc = nn.Sequential(
                nn.Linear(feats_dim, hidden_dim),
                nn.Dropout(feat_drop),
                nn.ELU(),
            )
            nn.init.xavier_normal_(fc[0].weight, gain=1.414)
            self.fc_list.append(fc)
        
        self.mp = mp_enc
        self.sc = sc_enc
        self.contrast = contrast

    def forward_features(self, feats, mps, nei_index):
        h_all = [fc(feat) for fc, feat in zip(self.fc_list, feats)]
        z_mp = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)
        return z_mp, z_sc

    def forward(self, feats, pos, mps, nei_index):  # p a s
        z_mp, z_sc = self.forward_features(feats, mps, nei_index)
        
        if isinstance(self.contrast, ContrastDrop):
            z_mp2, z_sc2 = self.forward_features(feats, mps, nei_index)
            return self.contrast(z_mp, z_mp2, z_sc, z_sc2, pos)

        else:
            return self.contrast(z_mp, z_sc, pos)

    @torch.no_grad()
    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        return self.mp(z_mp, mps)
