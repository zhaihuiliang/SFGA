import torch
import torch.nn as nn
from torch_geometric.utils import degree
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from models.SFGA_utils import AE_FUSE


class SFGA(nn.Module):
    def __init__(self, cfg):
        super(SFGA, self).__init__()
        self.args = cfg
        gpu_num_ = cfg.gpu_num
        if gpu_num_ == -1:
            self.device = 'cpu'
        else:
            self.device = torch.device("cuda:" + str(gpu_num_) if torch.cuda.is_available() else "cpu")

        self.alpha = cfg.sfga.alpha
        self.beta = cfg.sfga.beta

        self.fuse_aemodel = AE_FUSE(cfg)
        self.criteria = nn.BCEWithLogitsLoss()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, optimizer, epoch):
        ano_labels = np.squeeze(data.labels.cpu().numpy())
        for innerepoch in range(self.args.sfga.inner_epochs):
            optimizer.zero_grad()

            node_reps, feat_recons = self.fuse_aemodel(data.features, data.adj_list)

            feat_rec_loss, featrecons_nodelevel_loss = self._loss_feat_recons(feat_recons, data.features)

            featreconslossToScore = featrecons_nodelevel_loss
            ad_scores_featrecons = np.array(featreconslossToScore.cpu().detach())
            ad_scores_featrecons[np.isnan(ad_scores_featrecons)] = 0.5  # incase nan

            emb_cons_loss = self.emb_sim_cons_loss(node_reps, data.adj_list)

            # alpha
            c, d = self.alpha, (1 - self.alpha)
            loss_all = c * feat_rec_loss + d * emb_cons_loss

            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()

            ad_scores_cons = self.latent_cons_ad_score(node_reps, data.adj_list)
            ad_scores_cons[np.isnan(ad_scores_cons)] = 0.5  # incase nan

            a,b=self.beta,(1 - self.beta)

            ad_scores = a * ad_scores_featrecons + b * ad_scores_cons

            ad_auroc = roc_auc_score(ano_labels, ad_scores)
            ad_auprc = average_precision_score(ano_labels, ad_scores, average='macro', pos_label=1, sample_weight=None)

            print('epoch:', epoch, 'inner:', innerepoch,
                  '  AUROC: ', round(ad_auroc, 6),
                  '  AUPRC: ', round(ad_auprc, 6),
                  )

        return loss_all, ad_auroc, ad_auprc

    def preprocess(self, data, cfg):
        idx_p_list = []
        node_num = data.features.shape[0]
        for adj in data.adj_list:
            idx_p_list_0 = []
            if self.args.dataset.sparse:
                A_degree = degree(adj._indices()[0], data.features.shape[0], dtype=int).tolist()
                out_node = adj._indices()[1]
            else:
                A_degree = degree(adj.to_sparse()._indices()[0], data.features.shape[0], dtype=int).tolist()
                out_node = adj.to_sparse()._indices()[1]
            for i in range(node_num):  #
                if A_degree[i] == 0:
                    neighbors = [i]
                else:
                    start = sum(A_degree[:i])
                    end = start + A_degree[i]
                    neighbors = out_node[start:end].tolist()
                idx_p_list_0.append(neighbors)

            idx_p_list.append(idx_p_list_0)

        features = [data.features.to(self.device) for _ in range(self.args.dataset.num_view)]
        adj_list = [adj.to(self.device) for adj in data.adj_list]

        data.idx_p_list = idx_p_list
        data.features = features
        data.adj_list = adj_list
        data.dataset_name = cfg.dataset.name

    def _loss_feat_recons(self, feat_recons, feat_ori):
        l = torch.nn.MSELoss(reduction='sum')
        node_num = feat_ori[0].shape[0]
        recons_feat = 0
        for i in range(self.args.dataset.num_view):
            recons_feat += l(feat_recons[i], feat_ori[i])
        recons_feat /= node_num

        feat_recons_nodelevel = 0
        for i in range(self.args.dataset.num_view):
            feat_recons_nodelevel += torch.norm(feat_recons[i] - feat_ori[i], p=2, dim=1)

        node_feat_recons_loss = feat_recons_nodelevel

        recons_loss = recons_feat

        return recons_loss, node_feat_recons_loss

    def emb_sim_cons_loss(self, node_emb, adj_lists):
        global_cons_loss = 0
        for i in range(self.args.dataset.num_view):
            global_cons_loss += self.global_sim_cons(node_emb[i], adj_lists[i])

        local_cons_loss = 0
        for i in range(self.args.dataset.num_view):
            local_cons_loss += self.local_sim_cons(node_emb[i], adj_lists[i])

        sim_cons_loss = local_cons_loss + global_cons_loss
        return sim_cons_loss

    def local_sim_cons(self, node_emb, adj_matrix):
        feature_toNorm = torch.norm(node_emb, dim=-1, keepdim=True)
        normed_feature = node_emb / feature_toNorm
        normed_feature = torch.nan_to_num(normed_feature, nan=0.0, posinf=1e10, neginf=-1e10)
        sim_matrix = torch.mm(normed_feature, normed_feature.T)

        sim_matrix = torch.squeeze(sim_matrix) * adj_matrix

        sim_matrix[torch.isinf(sim_matrix)] = 0
        sim_matrix[torch.isnan(sim_matrix)] = 0

        row_sum = torch.sum(adj_matrix, 0)
        everynode_degree = row_sum
        inv_nodedegree = torch.pow(everynode_degree, -1).flatten()
        r_inv = inv_nodedegree

        r_inv[torch.isinf(r_inv)] = 0.

        everynode_sim_sum = torch.sum(sim_matrix, 1)

        everynode_sim_avg = everynode_sim_sum * r_inv
        allnode_sim_sum = torch.sum(everynode_sim_avg)

        node_level_loss = - allnode_sim_sum / node_emb.shape[0]

        return node_level_loss

    def global_sim_cons(self, node_emb, adj):
        node_emb = node_emb / torch.norm(node_emb, dim=-1, keepdim=True)
        node_emb = torch.nan_to_num(node_emb, nan=0.0, posinf=1e10, neginf=-1e10)
        sim_node_to_node = torch.mm(node_emb, node_emb.T)
        adj_inverse = (1 - adj)
        sim_not_nei_node = sim_node_to_node * adj_inverse
        sim_not_nei_node_sum = torch.sum(sim_not_nei_node, 1)
        row_sum = torch.sum(adj_inverse, 1)
        r_inv = torch.pow(row_sum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        sim_not_nei_node_avg = sim_not_nei_node_sum * r_inv
        global_loss = torch.sum(sim_not_nei_node_avg)
        node_level_global_loss = global_loss / node_emb.shape[0]
        return node_level_global_loss

    def latent_cons_ad_score(self, node_reps, adj_lists):
        cons_score = 0
        for i in range(self.args.dataset.num_view):
            cons_score += self.latent_sim_cons_cal(node_reps[i], adj_lists[i])
        cons_score = torch.unsqueeze(cons_score, 0)
        cons_score = np.array(torch.squeeze(cons_score).cpu().detach())
        cons_score = 1 - self.normalize_score(cons_score)
        ad_scores_cons = np.array(cons_score)
        return ad_scores_cons

    def latent_sim_cons_cal(self, feature, adj_matrix):
        feature = feature / torch.norm(feature, dim=-1, keepdim=True)
        feature = torch.nan_to_num(feature, nan=0.0, posinf=1e10, neginf=-1e10)
        sim_matrix = torch.mm(feature, feature.T)
        sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
        row_sum = torch.sum(adj_matrix, 0)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        cons_ad_score = torch.sum(sim_matrix, 1)
        cons_ad_score = cons_ad_score * r_inv
        return cons_ad_score

    def normalize_score(self, ano_score):
        ano_score = ((ano_score - np.min(ano_score)) / (
                np.max(ano_score) - np.min(ano_score)))
        return ano_score
