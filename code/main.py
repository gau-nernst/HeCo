import datetime
import pickle as pkl
import random
import warnings
from copy import deepcopy
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from module import HeCo, HeCoDrop
from utils import evaluate, load_data, set_params


warnings.filterwarnings('ignore')


# https://github.com/liun-online/HeCo/issues/1
def evaluate_cluster(embeds, y, n_label):
    Y_pred = KMeans(n_label, random_state=0).fit_predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    print(f"\t[clustering] nmi: {nmi:.4f} ari: {ari:.4f}")
    return nmi, ari


## random seed ##
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_data(args):
    device = args.device
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
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


def train(args, data):
    device = args.device
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = data
    P = int(len(mps))
    feats_dim_list = [i.shape[1] for i in feats]

    if args.heco_drop:
        model_cls = HeCoDrop
        inputs = (feats, pos, mps, nei_index, args.beta1, args.beta2, args.dcl)

    else:
        model_cls = HeCo
        inputs = (feats, pos, mps, nei_index)

    model = model_cls(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, args.nei_num, args.tau, args.lam).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    cnt_wait = 0
    best = 1e9
    best_t = 0
    best_state_dict = None

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model(*inputs)
        if epoch % 100 == 0:
            log_msg = (
                f"Epoch: {epoch:04d}, "
                f"Loss: {loss.cpu().item():.4f}, "
                f"mp: {model.mp.att.beta.round(4)}, "
                f"sc: {model.sc.inter.beta.round(4)}"
            )
            print(log_msg)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            best_state_dict = deepcopy(model.state_dict())
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(best_state_dict)
    model.eval()
    embeds = model.get_embeds(feats, mps)

    if args.save_emb:
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()

    return embeds


def eval(args, data, embeds):
    device = args.device
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = data
    nb_classes = label.shape[-1]

    metrics = {}
    for i in range(len(idx_train)):
        f1_macro, f1_micro, auc = evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
        metrics[f"f1_macro_{args.ratio[i]}"] = f1_macro
        metrics[f"f1_micro_{args.ratio[i]}"] = f1_micro
        metrics[f"auc_{args.ratio[i]}"] = auc

    embeds = embeds.cpu().data.numpy()
    label = np.argmax(label.cpu().data.numpy(), axis=-1)
    nmi, ari = evaluate_cluster(embeds, label, nb_classes)
    metrics["nmi"] = nmi
    metrics["ari"] = ari

    return metrics


def log_to_file(args, start_time, end_time, metrics):
    params = vars(args)
    log_data = [start_time, end_time]
    log_data.extend(params[k] for k in sorted(params.keys()))
    log_data.extend(metrics[k] for k in sorted(metrics.keys()))
    log_str = ",".join(f"\"{str(x)}\"" if isinstance(x, list) else str(x) for x in log_data) + "\n"
    with open(f"{args.dataset}.log", "a") as f:
        f.write(log_str)


def run(args):
    start_time = int(time.time())

    seed_everything(args.seed)
    data = get_data(args)
    embeds = train(args, data)
    metrics = eval(args, data, embeds)

    end_time = int(time.time())
    log_to_file(args, start_time, end_time, metrics)


if __name__ == '__main__':
    args = set_params()
    run(args)
