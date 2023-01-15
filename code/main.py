import argparse
import random
from copy import deepcopy
import itertools

import numpy as np
import torch
from tqdm import tqdm
import wandb

from module import HeCo, Mp_encoder, Sc_encoder, build_contrast, build_loss
from utils import evalulate_embeddings, load_data, set_params


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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
    if args.seed is not None:
        seed_everything(args.seed)

    logger = None
    if not args.disable_logging:
        logger = wandb.init(
            project="heco",
            name=f"{args.dataset}_{args.log_name}",
            config=vars(args),
        )

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
        args.multi_positive,
        args.feat_drop,
        mp_enc,
        sc_enc,
        contrast,
    )
    model = model.to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.l2_coef,
        betas=(args.adam_beta1, 0.999)
    )


    best, best_t, best_state_dict = float("inf"), 0, None
    for epoch in tqdm(itertools.count(), dynamic_ncols=True):
        model.train()
        optimiser.zero_grad()
        loss = model(feats, pos, mps, nei_index)

        loss_item = loss.item()
        if logger is not None:
            log_dict = dict(loss=loss_item, lr=optimiser.param_groups[0]["lr"])
            for i, x in enumerate(model.mp.att.beta):
                log_dict[f"mp/{i}"] = x
            for i, x in enumerate(model.sc.inter.beta):
                log_dict[f"sc/{i}"] = x
            logger.log(log_dict, step=epoch)

        if epoch % 100 == 0:
            embeds = model.get_embeds(feats, mps)
            metrics = evalulate_embeddings(args, embeds, label, idx_train, idx_val, idx_test, verbose=False)
            if logger is not None:
                logger.log(metrics, step=epoch)

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
    if logger is not None:
        logger.log(metrics, step=epoch) # should log best_t instead, but wandb doesn't allow writing to previous step
        logger.finish()


if __name__ == '__main__':
    args = set_params()
    main(args)
