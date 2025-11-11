import time
from tqdm import tqdm

from config import cfg, update_cfg, print_cfg
from trainer import Trainer
from utils.utils import set_seed

ad_auroc_list = []
ad_auprc_list = []

import os
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

def main(cfg):
    seeds = cfg.seed if cfg.seed is not None else range(cfg.runs)

    start = time.time()
    for seed in tqdm(seeds):
        set_seed(seed)
        cfg.defrost()
        cfg.seed = seed

        torch.cuda.empty_cache()

        trainer = Trainer(cfg)
        cfg.freeze()
        (_,ad_auroc_mean,
         ad_auprc_mean,
         )=trainer.train()

        ad_auroc_list.append(ad_auroc_mean)
        ad_auprc_list.append(ad_auprc_mean)

        del trainer
    end = time.time()


    print(f"Total Running time: {(end - start) / len(seeds):.2f}s")


if __name__ == '__main__':
    ### Grid Search
    cfg_list = update_cfg(cfg)
    for cfg in cfg_list:
        print_cfg(cfg)
        main(cfg)

    print("\t[SFGA anomaly detection] AUROC | AUPRC")
    print(f"\t&{(100 * max(ad_auroc_list)):.2f}  "
          f"  &{(100 * ad_auprc_list[ad_auroc_list.index(max(ad_auroc_list))]):.2f}"
          )



