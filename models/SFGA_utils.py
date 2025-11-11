import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, drop_prob, isBias=False):
        super(GCN, self).__init__()
        self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)
        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.drop_prob = drop_prob
        self.isBias = isBias

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj, sparse=False):
        feat = F.dropout(feat, self.drop_prob, training=self.training)
        feat_raw = self.fc_1(feat)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(feat_raw, 0)), 0)
        else:
            seq = torch.mm(adj, feat_raw)
        if self.isBias:
            seq += self.bias_1
        return self.act(seq)


class GNNEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.GCN = GCN(cfg.dataset.ft_size, cfg.gnn.hidden_dim, cfg.gnn.activation, cfg.gnn.dropout, cfg.gnn.isBias)
        self.linear_S = nn.Linear(cfg.gnn.hidden_dim, cfg.sfga.emb_dim)

    def forward(self, x, adj):
        tmp = self.GCN(x, adj)
        emb = self.linear_S(tmp)
        return emb


class Linearlayer(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Linearlayer, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                h = self.batch_norms[layer](h)
                h = F.relu(h)
            return self.linears[self.num_layers - 1](h)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear1 = Linearlayer(
            num_layers=cfg.sfga.decolayer,
            input_dim=cfg.sfga.emb_dim,
            hidden_dim=cfg.gnn.hidden_dim,
            output_dim=cfg.dataset.ft_size)
        self.linear2 = nn.Linear(cfg.dataset.ft_size, cfg.dataset.ft_size)

    def forward(self, s):
        recons = self.linear1(s)
        recons=F.relu(recons)
        recons = self.linear2(recons)
        return recons


class AE_FUSE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_view = cfg.dataset.num_view
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for _ in range(self.num_view):
            self.encoder.append(GNNEncoder(cfg))
            self.decoder.append(Decoder(cfg))

        self.cross_attn = CrossLayerAttention(num_layers=cfg.dataset.num_view, hidden_dim=cfg.sfga.emb_dim)

    def encode(self, x, adj_list):
        viewembs = []
        for i in range(self.num_view):
            viewembs.append(self.encoder[i](x[i], adj_list[i]))
            if not hasattr(self, '_memory_optimized'):
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                torch.cuda.empty_cache()
                self._memory_optimized = True

        enhanced_viewembs = []
        for view_idx in range(self.num_view):
            fused = self.cross_attn([viewembs[view_idx]] + viewembs[:view_idx] + viewembs[view_idx + 1:])
            enhanced_viewembs.append(fused)
        return enhanced_viewembs
    def decode(self, s):
        recons = []
        for i in range(self.num_view):
            r = self.decoder[i](s[i])
            recons.append(r)
        return recons

    def forward(self, x, adj):
        en_viewembs = self.encode(x, adj)
        recons = self.decode(en_viewembs)
        return en_viewembs, recons

class CrossLayerAttention(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, layer_embeddings):
        q = self.query(layer_embeddings[0])
        keys = [k(emb) for k, emb in zip(self.keys, layer_embeddings)]
        view_similarities = [torch.matmul(q, k.t()) for k in keys]
        attn_scores = torch.stack(view_similarities)
        mean_attn_scores = attn_scores.mean(dim=2)
        attn_weights = self.softmax(mean_attn_scores)
        fused = torch.stack([emb * w.unsqueeze(-1) for emb, w in zip(layer_embeddings, attn_weights)]).sum(0)
        return fused












