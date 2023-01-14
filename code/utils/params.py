import argparse
from typing import Any, Dict

import torch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["acm", "dblp", "freebase", "aminer"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--nb_epochs", type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument("--eva_lr", type=float)
    parser.add_argument("--eva_wd", type=float)

    # The parameters of learning process
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--l2_coef", type=float)
    parser.add_argument("--adam_beta1", type=float, default=0.9)

    # model parameter
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--feat_drop", type=float)
    parser.add_argument("--attn_drop", type=float)
    parser.add_argument("--sample_rate", nargs="+", type=int)   # used in sc_encoder

    # contrastive learning
    parser.add_argument("--multi_positive", action="store_true")
    parser.add_argument("--loss_type", default="info_nce", choices=["info_nce", "dcl", "arcface", "triplet"])
    parser.add_argument("--contrast_type", default="contrast", choices=["contrast", "contrast_drop"])
    parser.add_argument("--temp", type=float)
    parser.add_argument("--margin", type=float, default=0.0)    # for arcface and triplet loss
    parser.add_argument("--beta1", type=float, default=0.25)
    parser.add_argument("--beta2", type=float, default=0.25)

    parser.add_argument("--disable_logging", action="store_true")
    parser.add_argument("--log_name")

    return parser


def get_default_params(dataset: str) -> Dict[str, Any]:
    default_params = dict(
        acm=dict(
            # seed=0,
            eva_lr=0.05,
            eva_wd=0,
            patience=5,
            lr=8e-4,
            l2_coef=0,
            temp=0.8,
            feat_drop=0.3,
            attn_drop=0.5,
            sample_rate=[7, 1],
        ),
        dblp=dict(
            # seed=53,
            eva_lr=0.01,
            eva_wd=0,
            patience=30,
            lr=0.0008,
            l2_coef=0,
            temp=0.9,
            feat_drop=0.4,
            attn_drop=0.35,
            sample_rate=[6],
        ),
        aminer=dict(
            # seed=4,
            eva_lr=0.01,
            eva_wd=0,
            patience=40,
            lr=0.003,
            l2_coef=0,
            temp=0.5,
            feat_drop=0.5,
            attn_drop=0.5,
            sample_rate=[3, 8],
        ),
        freebase=dict(
            # seed=32,
            eva_lr=0.01,
            eva_wd=0,
            patience=20,
            lr=0.001,
            l2_coef=0,
            temp=0.5,
            feat_drop=0.1,
            attn_drop=0.3,
            sample_rate=[1, 18, 2],
        ),
    )
    return default_params[dataset]


def set_params() -> argparse.Namespace:
    args = get_parser().parse_args()
    default_params = get_default_params(args.dataset)

    for k, v in default_params.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    return args
