import numpy as np
import torch
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, f1_score,
                             normalized_mutual_info_score, roc_auc_score)
from torch import nn

from .load_data import _RATIOS


class LogReg(nn.Linear):
    def __init__(self, embed_dim, num_classes):
        super().__init__(embed_dim, num_classes)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)


def evalulate_embeddings(args, embeds, label, idx_train, idx_val, idx_test, verbose=True):
    metrics = {}
    for ratio, i_train, i_val, i_test in zip(_RATIOS, idx_train, idx_val, idx_test):
        f1_macro, f1_micro, auc = evaluate_classification(embeds, i_train, i_val, i_test, label,
                 args.eva_lr, args.eva_wd, verbose=verbose)
        metrics[f"f1_macro_{ratio}"] = f1_macro
        metrics[f"f1_micro_{ratio}"] = f1_micro
        metrics[f"auc_{ratio}"] = auc

    embeds = embeds.cpu().numpy()
    label = np.argmax(label.cpu().numpy(), axis=-1)
    nmi, ari = evaluate_clustering(embeds, label, verbose=verbose)
    metrics["nmi_l2"] = nmi
    metrics["ari_l2"] = ari

    embeds = normalize(embeds)
    nmi, ari = evaluate_clustering(embeds, label, verbose=verbose)
    metrics["nmi_cosine"] = nmi
    metrics["ari_cosine"] = ari

    return metrics


def evaluate_classification(embeds, idx_train, idx_val, idx_test, label, lr, wd, verbose=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(20):
        log = LogReg(hid_units, label.shape[-1]).to(embeds.device)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            with torch.no_grad():
                logits = log(test_embs)
                preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = torch.softmax(best_logits, dim=1)
        auc_score = roc_auc_score(test_lbls.cpu(), best_proba.cpu(), multi_class='ovr')
        auc_score_list.append(auc_score)

    macro_f1_mean, macro_f1_std = np.mean(macro_f1s), np.std(macro_f1s)
    micro_f1_mean, micro_f1_std = np.mean(micro_f1s), np.std(micro_f1s)
    auc_mean, auc_std = np.mean(auc_score_list), np.std(auc_score_list)
    if verbose:
        msg = (
            "\t[Classification] "
            f"Macro-F1: {macro_f1_mean * 100:.4f} std: {macro_f1_std * 100:.4f} "
            f"Micro-F1: {micro_f1_mean * 100:.4f} std: {micro_f1_std * 100:.4f} "
            f"AUC: {auc_mean * 100:.4f} std: {auc_std * 100:.4f}"
        )
        print(msg)
    return macro_f1_mean, micro_f1_mean, auc_mean


# https://github.com/liun-online/HeCo/issues/1
def evaluate_clustering(embeds, y, verbose=True):
    nmis, aris = [], []
    for _ in range(20):
        Y_pred = KMeans(y.max() + 1).fit_predict(embeds)
        nmi = normalized_mutual_info_score(y, Y_pred)
        ari = adjusted_rand_score(y, Y_pred)
        nmis.append(nmi)
        aris.append(ari)
    nmi_mean, nmi_std = np.mean(nmis), np.std(nmis, ddof=1)
    ari_mean, ari_std = np.mean(aris), np.std(aris, ddof=1)
    if verbose:
        print(f"\t[Clustering] NMI: {nmi_mean*100:.4f} std: {nmi_std*100:.4f} ARI: {ari_mean*100:.4f} std: {ari_std*100:.4f}")
    return nmi_mean, ari_mean
