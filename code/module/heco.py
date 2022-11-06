import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrast import build_contrast
from .losses import build_loss
from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder


class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, temp, loss_type, contrast_type, beta1, beta2):
        super(HeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.contrast_type = contrast_type
        self.fc_list = nn.ModuleList()
        for feats_dim in feats_dim_list:
            fc = nn.Sequential(
                nn.Linear(feats_dim, hidden_dim),
                nn.Dropout(feat_drop),
                nn.ELU(),
            )
            nn.init.xavier_normal_(fc[0].weight, gain=1.414)
            self.fc_list.append(fc)
        
        self.mp = Mp_encoder(P, hidden_dim, attn_drop)
        self.sc = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)

        loss = build_loss(loss_type, temp, 0)
        self.contrast = build_contrast(contrast_type, hidden_dim, loss, beta1, beta2)

    def forward_features(self, feats, mps, nei_index):
        h_all = [fc(feat) for fc, feat in zip(self.fc_list, feats)]
        z_mp = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)
        return z_mp, z_sc

    def forward(self, feats, pos, mps, nei_index):  # p a s
        z_mp, z_sc = self.forward_features(feats, mps, nei_index)
        
        if self.contrast_type == "contrast_drop":
            z_mp2, z_sc2 = self.forward_features(feats, mps, nei_index)
            return self.contrast(z_mp, z_mp2, z_sc, z_sc2, pos)

        else:
            return self.contrast(z_mp, z_sc, pos)

    @torch.no_grad()
    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        return self.mp(z_mp, mps)
