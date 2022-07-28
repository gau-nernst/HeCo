import datetime
import pickle as pkl
import random
import warnings
from copy import deepcopy

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from module import HeCo, HeCoDrop
from utils import evaluate, load_data, set_params


# https://github.com/liun-online/HeCo/issues/1
def evaluate_cluster(embeds, y, n_label):
    Y_pred = KMeans(n_label, random_state=0).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    return nmi, ari

def evaluate_clu(dataset, embeds, label):
    if dataset =="acm":
        n_label = 3
    elif dataset =="dblp":
        n_label = 4
    elif dataset == "aminer":
        n_label = 4
    elif dataset == "freebase":
        n_label = 3
    embeds = embeds.cpu().data.numpy()
    label = np.argmax(label.cpu().data.numpy(), axis=-1)
    nmi, ari = evaluate_cluster(embeds, label, n_label)
    print("\t[clustering] nmi: {:.4f} ari: {:.4f}"
            .format(nmi, ari) )


warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## random seed ##
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train():
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    print("seed ",args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)
    
    # move data to device e.g. GPU
    feats = [feat.to(device) for feat in feats]
    mps = [mp.to(device) for mp in mps]
    pos = pos.to(device)
    label = label.to(device)
    idx_train = [i.to(device) for i in idx_train]
    idx_val = [i.to(device) for i in idx_val]
    idx_test = [i.to(device) for i in idx_test]

    if args.heco_drop:
        model_cls = HeCoDrop
        inputs = (feats, pos, mps, nei_index, args.beta1, args.beta2)

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
        
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(best_state_dict)
    model.eval()
    embeds = model.get_embeds(feats, mps)
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    
    evaluate_clu(args.dataset, embeds, label)

    if args.save_emb:
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()


if __name__ == '__main__':
    train()
