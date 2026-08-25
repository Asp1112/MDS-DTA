import os
import pickle
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None, force_reprocess=False):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        p = self.processed_paths[0]
        if force_reprocess:
            print('Force re-process enabled for {}'.format(p))
            self._process()
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(p)
        elif os.path.isfile(p) and os.path.getsize(p) > 0:
            print('Pre-processed data found: {}, loading ...'.format(p))
            try:
                self.data, self.slices = torch.load(p)
            except (EOFError, RuntimeError, pickle.UnpicklingError):
                if xd is not None and xt is not None and y is not None and smile_graph is not None:
                    print('Corrupt pre-processed file detected: {}, rebuilding ...'.format(p))
                    self._process()
                    self.process(xd, xt, y, smile_graph)
                    self.data, self.slices = torch.load(p)
                else:
                    raise
        else:
            if xd is not None and xt is not None and y is not None and smile_graph is not None:
                print('Pre-processed data {} not found, doing pre-processing...'.format(p))
                self._process()
                self.process(xd, xt, y, smile_graph)
                self.data, self.slices = torch.load(p)
            else:
                raise FileNotFoundError(p)

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        tmp_path = self.processed_paths[0] + '.tmp'
        torch.save((data, slices), tmp_path)
        os.replace(tmp_path, self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y, f):
    y = np.asarray(y).ravel()
    f = np.asarray(f).ravel()
    if y.size == 0:
        return np.nan
    ind = np.argsort(y, kind='mergesort')
    y_sorted = y[ind]
    f_sorted = f[ind]

    f_unique = np.unique(f_sorted)
    ranks = np.searchsorted(f_unique, f_sorted, side='left')

    m = f_unique.size
    bit = np.zeros(m + 1, dtype=np.int64)
    freq = np.zeros(m, dtype=np.int64)

    def bit_update(idx, delta=1):
        i = idx + 1
        while i <= m:
            bit[i] += delta
            i += i & -i

    def bit_query(idx):
        if idx < 0:
            return 0
        s = 0
        i = idx + 1
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    uniq_y, counts = np.unique(y_sorted, return_counts=True)
    start = 0
    total_processed = 0
    comparable = 0
    concordant = 0
    ties = 0
    for c in counts:
        end = start + c
        group_ranks = ranks[start:end]
        prev = total_processed
        if prev > 0:
            comparable += prev * c
        for r in group_ranks:
            concordant += bit_query(r - 1)
            ties += freq[r]
        for r in group_ranks:
            bit_update(r, 1)
            freq[r] += 1
        total_processed += c
        start = end

    if comparable == 0:
        return np.nan
    return (concordant + 0.5 * ties) / comparable

def r2(y, f):
    """计算决定系数 R²"""
    ss_res = np.sum((y - f) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def mae(y, f):
    """
    计算平均绝对误差 MAE
    """
    mae = np.mean(np.abs(y - f))
    return mae

def rm2(y_true, y_pred):
    """Calculate Rm² metric."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2_val = r2(y_true, y_pred)

    # R0²: regression through origin
    slope = np.sum(y_true * y_pred) / np.sum(y_pred ** 2)
    y_pred_origin = slope * y_pred
    r2_origin = r2(y_true, y_pred_origin)

    rm2_val = r2_val * (1 - np.sqrt(abs(r2_val - r2_origin)))
    return rm2_val
