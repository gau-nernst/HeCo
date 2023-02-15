from torch import nn, Tensor


class Contrast(nn.Module):
    def __init__(self, hidden_dim: int, loss: nn.Module):
        super().__init__()
        self.loss = loss
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def forward(self, z_mp: Tensor, z_sc: Tensor, pos_mat: Tensor) -> Tensor:
        z_mp, z_sc = map(self.proj, (z_mp, z_sc))
        return self.loss(z_mp, z_sc, pos_mat=pos_mat)


class ContrastDrop(Contrast):
    def __init__(self, hidden_dim: int, loss: nn.Module, beta1: float, beta2: float):
        assert beta1 >= 0 and beta2 >= 0 and 0 <= beta1 + beta2 <= 1
        super().__init__(hidden_dim, loss)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta = 1.0 - beta1 - beta2

    def forward(self, z_mp1: Tensor, z_mp2: Tensor, z_sc1: Tensor,  z_sc2: Tensor, pos_mat: Tensor) -> Tensor:
        z_mp1, z_mp2, z_sc1, z_sc2 = map(self.proj, (z_mp1, z_mp2, z_sc1, z_sc2))

        mp_sc_loss = self.loss(z_mp1, z_sc1, pos_mat=pos_mat)
        mp_loss = self.loss(z_mp1, z_mp2, pos_mat=pos_mat, two_way=False)
        sc_loss = self.loss(z_sc1, z_sc2, pos_mat=pos_mat, two_way=False)
        return mp_sc_loss * self.beta + mp_loss * self.beta1 + sc_loss * self.beta2


def build_contrast(contrast_type: str, hidden_dim: int, loss: nn.Module, beta1: float, beta2: float):
    contrast_mapping = dict(
        contrast=Contrast,
        contrast_drop=ContrastDrop,
    )
    assert contrast_type in contrast_mapping
    kwargs = dict(hidden_dim=hidden_dim, loss=loss)
    if contrast_type == "contrast_drop":
        kwargs.update(beta1=beta1, beta2=beta2)
    return contrast_mapping[contrast_type](**kwargs)
