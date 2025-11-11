import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import os
import os.path as osp
from collections.abc import Sequence
from typing import Any, Callable, List
import errno
class Dataset_ad(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.args = cfg
        gpu_num_ = cfg.gpu_num
        if gpu_num_ == -1:
            self.device = 'cpu'
        else:
            self.device = torch.device("cuda:" + str(gpu_num_) if torch.cuda.is_available() else "cpu")

        self._download_ad()
        self._process_ad()

        dataset = torch.load(self.processed_paths[0], map_location=self.device)
        self.features = dataset['features'].to(self.device)
        self.labels = dataset['labels'].to(self.device)
        self.adj_list = dataset['adj_list']

        self.ft_size = self.features.shape[1]
        self.nb_nodes = self.adj_list[0].shape[1]
        self.num_view = len(self.adj_list)

        # update dataset cfgs
        cfg.dataset.nb_nodes = self.nb_nodes
        cfg.dataset.ft_size = self.ft_size
        cfg.dataset.num_view = self.num_view
        cfg.dataset.name=dataset['dataset_name']

    @property
    def raw_dir(self):
        return os.path.join(self.args.dataset.root, self.args.dataset_name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.args.dataset.root, self.args.dataset_name, 'processed')

    @property
    def raw_file_names(self):
        if self.args.dataset_name.lower() == 'imdb_injected':
            return ['imdb_injected.mat']
        elif self.args.dataset_name.lower() == 'freebase_injected':
            return ['freebase_injected.mat']
        elif self.args.dataset_name.lower() == 'amazon_fraud':
            return ['Amazon_fraud.mat']
        elif self.args.dataset_name.lower() == 'yelpchi_fraud':
            return ['YelpChi_fraud.mat']

    @property
    def processed_file_names(self):
        if self.args.dataset_name.lower() == 'imdb_injected':
            return ['imdb_injected.pt']
        elif self.args.dataset_name.lower() == 'freebase_injected':
            return ['freebase_injected.pt']
        elif self.args.dataset_name.lower() == 'amazon_fraud':
            return ['Amazon_fraud.pt']
        elif self.args.dataset_name.lower() == 'yelpchi_fraud':
            return ['YelpChi_fraud.pt']

    @property
    def meta_path_names(self):
        if self.args.dataset_name.lower() == 'imdb_injected':
            return ["MAM_INJ","MDM_INJ"]
        elif self.args.dataset_name.lower() == 'freebase_injected':
            return ["MAM_INJ","MDM_INJ","MWM_INJ"]
        elif self.args.dataset_name.lower() == "amazon_fraud":
            return ["net_upu", "net_usu", "net_uvu"]
        elif self.args.dataset_name.lower() == "yelpchi_fraud":
            return ["net_rur", "net_rtr", "net_rsr"]

    @property
    def raw_paths(self):
        files = self.raw_file_names
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.raw_dir, f) for f in to_list(files)]

    @property
    def processed_paths(self):
        files = self.processed_file_names
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.processed_dir, f) for f in to_list(files)]
    def _download_ad(self):
        if files_exist(self.raw_paths):
            return
    def _process_ad(self):
        if files_exist(self.processed_paths):
            return
        makedirs(self.processed_dir)

        ### load raw data
        if self.args.dataset_name.lower() in ["imdb_injected","freebase_injected"]:
            meta_paths = self.meta_path_names
            adj_list, features, labels = self.load_data_ad(self.raw_paths, meta_paths, self.args.dataset.sc)
            features = features.todense()
            features=features+1e-10 # in case nan

        elif self.args.dataset_name.lower() in ["amazon_fraud","yelpchi_fraud"]:
            meta_paths = self.meta_path_names
            adj_list, features, labels = self.load_data_ad(self.raw_paths, meta_paths, self.args.dataset.sc)
            features = preprocess_features(features)

        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(labels)
        num_nodes = features.shape[0]

        ### transform and normalize adjacency matrix
        adj_list = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        adj_list = [adj.to_dense() for adj in adj_list]
        adj_list = [normalize_graph(adj) for adj in adj_list]

        ### Save to .pt
        dataset = {
            'features': features,
            'labels': labels,
            'adj_list': adj_list,
            'dataset_name': self.args.dataset_name.lower()
        }
        torch.save(dataset, self.processed_paths[0])

    def __len__(self):
        return self.nb_nodes

    def __getitem__(self, idx):
        return self.tf_input_list[:, idx, :, :], self.pos_attention_list[:, idx, :, :], self.padding_list[:, idx, :], \
        self.labels[idx]

    def load_data_ad(self, root, meta_paths, sc=3):
        ### load data file
        data = sio.loadmat(root[0])
        ### labels
        if self.args.dataset_name.lower() in ["imdb_injected","freebase_injected"]:
            label = data['Label']
        ### adj_list
        adj_list = []
        for meta_path in meta_paths:
            adj = data[meta_path] + np.eye(data[meta_path].shape[0]) * sc
            adj = sp.csr_matrix(adj)
            adj_list.append(adj)

        ### features
        if any(sub in root[0] for sub in ['Amazon_fraud', 'YelpChi_fraud']):
            truefeatures = data['features'].astype(float)
        else:
            truefeatures = data['Attributes'].astype(float)
        truefeatures = sp.lil_matrix(truefeatures)

        if "Amazon_fraud" in root[0] or "YelpChi_fraud" in root[0]:
            label=data['label']
        return adj_list, truefeatures, label

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def makedirs(path: str):
    r"""Recursively creates a directory.

    Args:
        path (str): The path to create.
    """
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def files_exist(files: List[str]) -> bool:
    # return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])