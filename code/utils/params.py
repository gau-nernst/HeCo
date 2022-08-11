import argparse
from typing import Any, Dict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--save_emb", action="store_true")
    parser.add_argument("--turn", type=int, default=0)
    parser.add_argument("--ratio", nargs="+", type=int, default=[20, 40, 60])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--nb_epochs", type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument("--eva_lr", type=float)
    parser.add_argument("--eva_wd", type=float)

    # The parameters of learning process
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--l2_coef", type=float)

    # model-specific parameters
    parser.add_argument("--tau", type=float)
    parser.add_argument("--feat_drop", type=float)
    parser.add_argument("--attn_drop", type=float)
    parser.add_argument("--sample_rate", nargs="+", type=int)
    parser.add_argument("--lam", type=float)

    # HeCo-drop
    parser.add_argument("--heco_drop", action="store_true")
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)

    # decoupled contrastive loss
    parser.add_argument("--dcl", action="store_true")

    return parser


def get_default_params(dataset: str) -> Dict[str, Any]:
    if dataset == "acm":
        return {
            "seed": 0,
            "eva_lr": 0.05,
            "eva_wd": 0,
            "patience": 5,
            "lr": 0.0008,
            "l2_coef": 0,
            "tau": 0.8,
            "feat_drop": 0.3,
            "attn_drop": 0.5,
            "sample_rate": [7, 1],
            "lam": 0.5,
            "beta1": 0.5,
            "beta2": 0.5,
            "type_num": [4019, 7167, 60],
            "nei_num": 2,
        }

    if dataset == "dblp":
        return {
            "seed": 53,
            "eva_lr": 0.01,
            "eva_wd": 0,
            "patience": 30,
            "lr": 0.0008,
            "l2_coef": 0,
            "tau": 0.9,
            "feat_drop": 0.4,
            "attn_drop": 0.35,
            "sample_rate": [6],
            "lam": 0.5,
            "beta1": 0.5,
            "beta2": 0.5,
            "type_num": [4057, 14328, 7723, 20],
            "nei_num": 1,
        }

    if dataset == "aminer":
        return {
            "seed": 4,
            "eva_lr": 0.01,
            "eva_wd": 0,
            "patience": 40,
            "lr": 0.003,
            "l2_coef": 0,
            "tau": 0.5,
            "feat_drop": 0.5,
            "attn_drop": 0.5,
            "sample_rate": [3, 8],
            "lam": 0.5,
            "beta1": 0.5,
            "beta2": 0.5,
            "type_num": [6564, 13329, 35890],
            "nei_num": 2,
        }

    if dataset == "freebase":
        return {
            "seed": 32,
            "eva_lr": 0.01,
            "eva_wd": 0,
            "patience": 20,
            "lr": 0.001,
            "l2_coef": 0,
            "tau": 0.5,
            "feat_drop": 0.1,
            "attn_drop": 0.3,
            "sample_rate": [1, 18, 2],
            "lam": 0.5,
            "beta1": 0.5,
            "beta2": 0.5,
            "type_num": [3492, 2502, 33401, 4459],
            "nei_num": 3,
        }

    raise ValueError(f"Unknown dataset: {dataset}")


def set_params() -> argparse.Namespace:
    args = get_parser().parse_args()
    default_params = get_default_params(args.dataset)

    for k, v in default_params.items():
        if k in ("type_num", "nei_num"):
            setattr(args, k, v)

        elif getattr(args, k) is None:
            setattr(args, k, v)

    if not args.heco_drop:
        args.beta1 = None
        args.beta2 = None
    
        if args.dcl:
            raise NotImplementedError("DCL is not implemented for pure HeCo")

    return args
