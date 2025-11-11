import torch
import time
from utils.utils import set_seed
from utils.data_process_ad import Dataset_ad
import numpy as np
import os

class Trainer():
    def __init__(self, cfg):
        set_seed(cfg.seed)
        # Init args
        gpu_num_ = cfg.gpu_num
        if gpu_num_ == -1:
            self.device = 'cpu'
        else:
            self.device = torch.device("cuda:" + str(gpu_num_) if torch.cuda.is_available() else "cpu")
        self.dataset_name = cfg.dataset_name
        self.model_name = cfg.model_name
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.epochs = cfg.epochs
        self.use_batch = cfg.use_batch
        # Init Data
        self.data_ad=Dataset_ad(cfg)

        # Init Model
        from models import SFGA
        self.model = SFGA(cfg).to(self.device)
        # Init Optimizer 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.preprocess(self.data_ad,cfg)
        cfg.freeze()

    def train(self):
        print("Started training...") 
        start = time.time()
        ad_auroc_list=[]
        ad_auprc_list=[]

        import torch
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        torch.cuda.empty_cache()

        for epoch in range(self.epochs):
            if epoch % 5 == 0:
                torch.cuda.empty_cache()

            ### Train
            self.model.train()

            loss, ad_auroc, ad_auprc = self.model(self.data_ad, self.optimizer, epoch)
            train_loss = loss.item()
            ad_auroc_list.append(ad_auroc)
            ad_auprc_list.append(ad_auprc)

            ### Validate
            self.model.eval()
            with torch.no_grad():
                val_loss = train_loss

        end = time.time()

        print("\t[SFGA anomaly detection] AUROC | AUPRC")
        print(f"\t&{(100 * np.mean(ad_auroc_list)):.2f}±{(100 * np.std(ad_auroc_list)):.2f}  "
              f"  &{(100 * np.mean(ad_auprc_list)):.2f}±{(100 * np.std(ad_auprc_list)):.2f}")


        return (self.model,np.mean(ad_auroc_list),np.mean(ad_auprc_list),)

