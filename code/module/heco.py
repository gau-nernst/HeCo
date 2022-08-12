import torch
import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder
from .contrast import Contrast, ContrastDrop


class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, dcl, heco_drop, beta1, beta2):
        super(HeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList(nn.Linear(feats_dim, hidden_dim)
                                     for feats_dim in feats_dim_list)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else nn.Identity()
        self.mp = Mp_encoder(P, hidden_dim, attn_drop)
        self.sc = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)

        # projection head, only used in training
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        self.dcl = dcl
        self.heco_drop = heco_drop
        if not heco_drop:
            self.contrast = Contrast(dcl, tau, lam)
        else:
            self.contrast = ContrastDrop(dcl, tau, lam, beta1, beta2)

    def forward_features(self, feats, mps, nei_index):
        h_all = [F.elu(self.feat_drop(fc(feat))) for fc, feat in zip(self.fc_list, feats)]
        z_mp = self.proj(self.mp(h_all[0], mps))
        z_sc = self.proj(self.sc(h_all, nei_index))
        return z_mp, z_sc

    def forward(self, feats, pos, mps, nei_index):  # p a s
        if not self.heco_drop:
            z_mp, z_sc = self.forward_features(feats, mps, nei_index)
            loss = self.contrast(z_mp, z_sc, pos)

        else:
            z_mp1, z_sc1 = self.forward_features(feats, mps, nei_index)
            z_mp2, z_sc2 = self.forward_features(feats, mps, nei_index)
            loss = self.contrast(z_mp1, z_mp2, z_sc1, z_sc2, pos)

        return loss

    @torch.inference_mode()
    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        return self.mp(z_mp, mps)
