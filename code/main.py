import argparse
import random
from copy import deepcopy
import itertools

import numpy as np
import torch
from tqdm import tqdm
import wandb
from sklearn.cluster import KMeans, spectral_clustering
import torch.nn.functional as F
import scipy.sparse
import networkx

from module import HeCo, Mp_encoder, Sc_encoder, build_contrast, build_loss
from utils import evalulate_embeddings, load_data, set_params


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.autograd.set_detect_anomaly(True)


def seed_everything(seed):
    if seed < 0:
        print("Seed is negative. Seed will not be applied")
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def run_kmeans(embs: torch.Tensor, n_clusters: int):
    kmeans = KMeans(n_clusters)
    embs_np = F.normalize(embs.detach()).cpu().numpy()  # normalize vectors so that Euclidean distance is equal to cosine distance
    assignments = kmeans.fit_predict(embs_np)
    for i in range(n_clusters):
        if (assignments == i).sum() == 0:
            raise RuntimeError(f"Cluster {i} has no members")
    assignments = torch.from_numpy(assignments).to(embs.device).long()
    centroids = torch.from_numpy(kmeans.cluster_centers_).to(embs.device)
    return assignments, centroids


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

    loss = build_loss(args)
    if args.loss_type == "spectral_clustering":
        adj = mps[0]
        for mp in mps[1:]:
            adj = adj + mp
        adj = adj.coalesce()
        values = adj.values().cpu().numpy()
        indices = adj.indices().cpu().numpy()
        adj = scipy.sparse.coo_matrix((values, (indices[0], indices[1])))

    #     adj = networkx.from_scipy_sparse_array(adj)
    #     for component in networkx.connected_components(adj):

        labels = spectral_clustering(adj, n_clusters=args.n_clusters)
        loss.labels = torch.from_numpy(labels).long().to(args.device)

    contrast = build_contrast(args.contrast_type, args.hidden_dim, loss, args.mp_beta, args.sc_beta)

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

    if args.save_embs is not None:
        model.train()
        with torch.no_grad():
            z_mp, z_sc = model.forward_features(feats, mps, nei_index)
        np.save(f"{args.dataset}_before_train_{args.save_embs}_mp", z_mp.cpu().numpy())
        np.save(f"{args.dataset}_before_train_{args.save_embs}_sc", z_sc.cpu().numpy())

        np.save(f"{args.dataset}_before_train_{args.save_embs}_mp1", model.get_embeds(feats, mps).cpu().numpy())
        np.save(f"{args.dataset}_before_train_{args.save_embs}_mp2", model.get_embeds(feats, mps).cpu().numpy())

    best, best_t, best_state_dict = float("inf"), 0, None
    for epoch in tqdm(itertools.count(), dynamic_ncols=True):
        if epoch % 100 == 0 and logger is not None:
            model.eval()
            embeds = model.get_embeds(feats, mps)
            metrics = evalulate_embeddings(args, embeds, label, idx_train, idx_val, idx_test, verbose=False)
            logger.log(metrics, step=epoch)

        if args.loss_type == "deepcluster" and epoch % args.cluster_interval == 0:
            model.eval()
            with torch.no_grad():
                embeds = model.get_embeds(feats, mps)
                assignments, centroids = run_kmeans(embeds, args.n_clusters)

                model.contrast.loss.assignments = assignments
                model.contrast.loss.centroids.copy_(centroids)

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
    
    if args.save_embs is not None:
        model.train()
        with torch.no_grad():
            z_mp, z_sc = model.forward_features(feats, mps, nei_index)
        np.save(f"{args.dataset}_{args.save_embs}_mp", z_mp.cpu().numpy())
        np.save(f"{args.dataset}_{args.save_embs}_sc", z_sc.cpu().numpy())

        np.save(f"{args.dataset}_{args.save_embs}_mp1", model.get_embeds(feats, mps).cpu().numpy())
        np.save(f"{args.dataset}_{args.save_embs}_mp2", model.get_embeds(feats, mps).cpu().numpy())

    model.eval()
    embeds = model.get_embeds(feats, mps)
    metrics = evalulate_embeddings(args, embeds, label, idx_train, idx_val, idx_test)
    if logger is not None:
        logger.log(metrics, step=epoch) # should log best_t instead, but wandb doesn't allow writing to previous step
        logger.finish()


if __name__ == '__main__':
    args = set_params()
    main(args)
