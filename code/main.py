import argparse
import random
from copy import deepcopy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from module import HeCo, Mp_encoder, Sc_encoder, build_contrast, build_loss
from utils import evalulate_embeddings, load_data, set_params


def seed_everything(seed):
    if seed < 0:
        print("Seed is negative. Seed will not be applied")
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_data(args):
    device = args.device
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = load_data(args.dataset)
    print("seed ",args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", len(mps))
    
    # move data to device e.g. GPU
    feats = [feat.to(device) for feat in feats]
    mps = [mp.to(device) for mp in mps]
    pos = pos.to(device)
    label = label.to(device)
    idx_train = [i.to(device) for i in idx_train]
    idx_val = [i.to(device) for i in idx_val]
    idx_test = [i.to(device) for i in idx_test]

    return nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test


def main(args: argparse.Namespace):
    logger = SummaryWriter()

    device = args.device
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = get_data(args)
    P = int(len(mps))
    feats_dim_list = [i.shape[1] for i in feats]

    mp_enc = Mp_encoder(P, args.hidden_dim, args.attn_drop)
    sc_enc = Sc_encoder(args.hidden_dim, args.sample_rate, args.attn_drop)

    loss = build_loss(args.loss_type, args.temp, args.margin)
    contrast = build_contrast(args.contrast_type, args.hidden_dim, loss, args.beta1, args.beta2)

    model = HeCo(
        args.hidden_dim,
        feats_dim_list,
        args.feat_drop,
        mp_enc,
        sc_enc,
        contrast,
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)


    best, best_t, best_state_dict = float("inf"), 0, None
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model(feats, pos, mps, nei_index)

        loss_item = loss.item()
        logger.add_scalar("loss", loss_item, epoch)
        logger.add_scalar("lr", optimiser.param_groups[0]["lr"], epoch)
        logger.add_scalars("mp", {str(i): x for i, x in enumerate(model.mp.att.beta)}, epoch)
        logger.add_scalars("sc", {str(i): x for i, x in enumerate(model.sc.inter.beta)}, epoch)

        if loss_item < best:
            best, best_t = loss_item, epoch
            best_state_dict = deepcopy(model.state_dict())

        elif epoch - best_t >= args.patience:
            print('Early stopping!')
            break
        
        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(best_state_dict)
    model.eval()
    embeds = model.get_embeds(feats, mps)

    metrics = evalulate_embeddings(args, embeds, label, idx_train, idx_val, idx_test)

    hparams = {k: str(v) if isinstance(v, list) else v for k, v in vars(args).items()}
    logger.add_hparams(hparams, metrics)


if __name__ == '__main__':
    args = set_params()
    main(args)
