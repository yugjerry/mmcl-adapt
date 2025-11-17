import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import sys
from matplotlib import pyplot as plt
import warnings
from scipy.linalg import qr, sqrtm
import seaborn as sns
from tqdm import tqdm
from pytorch_metric_learning import losses
from sklearn.decomposition import PCA
import skdim
import argparse
import math
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from skdim.id import MLE

import umap

from utils import *
from sympy import *

seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)

device = "cpu" if not torch.cuda.is_available() else torch.device("cuda")
print("Device is :: {}...".format(device))


sns.set_theme()
sns.set_context("notebook")

# plot
import seaborn as sns
sns.set_theme()
sns.set_context("notebook")
import matplotlib.pyplot as plt
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def top_k_err(X, Y, k):
    X_, Y_ = X, Y
    logits_mat = torch.matmul(X_,Y_.T)
    topk_list = [logits_mat[i,:].topk(k).indices for i in range(logits_mat.shape[0])]
    err = [i not in topk_list[i] for i in range(len(topk_list))]

    return np.mean(err)

def match_acc(X, Y, lab1, lab2):
    X_, Y_ = X, Y
    logits_mat = torch.matmul(X_,Y_.T)
    match_list1 = [logits_mat[i,:].topk(1).indices[0] for i in range(logits_mat.shape[0])]
    match_list2 = [logits_mat[:,j].topk(1).indices[0] for j in range(logits_mat.shape[1])]
#     acc = np.mean([lab1[i] == lab2[j] for i, j in zip(match_list1, match_list2)])
    acc_1 = np.array([lab1[i] == lab1[match_list1[i]] for i in range(len(match_list1))])
    acc_2 = np.array([lab2[i] == lab2[match_list2[i]] for i in range(len(match_list2))])
    acc = np.mean(acc_1 * acc_2)
    
    return acc_1, acc_2
    

def postprocess(i_rp, root_dir, dataset,joint,n,arch,link,outdim,K_frac,repN):
    
    if dataset == 'synthetic':
        d_x, d_y, d_z = 20, 20, 5
    elif dataset[:4] == 'yfcc' or dataset[:4] == 'imag':
        d_x, d_y, d_z = 1024, 768, 0
        if joint == False:
            d_x = 768
    elif dataset == 'citeseq':
        d_x, d_y, d_z = 200, 24, 0



       
    
    sub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}'
    subsub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}/file_{d_z}_{arch}_{link}_{repN}_{outdim}'
    
    file_dir = f'{subsub_root}/rep_{i_rp}_{n}_{d_x}_{d_y}_{d_z}_{dataset}_{outdim}_{joint}.npz'
    res = np.load(file_dir, allow_pickle=True)
    
    X_rep=res['X_rep']
    Y_rep=res['Y_rep']
    X_rep_test=res['X_rep_test']
    Y_rep_test=res['Y_rep_test']
    n_test = X_rep_test.shape[0]
    tau_seq=res['tau_seq']
    loss_clip=res['loss_clip']
    loss_align=res['loss_align']
    lab1_train = res['lab1_train']
    lab2_train = res['lab2_train']
    lab1_test = res['lab1_test']
    lab2_test = res['lab2_test']

    n, n_test = X_rep.shape[0], X_rep_test.shape[0]

    if outdim > 1:
        mle = MLE()
        if arch == 'linear':
            mle = MLE(K=5)
        X_dim = mle.fit_transform(X_rep_test)
        Y_dim = mle.fit_transform(Y_rep_test)
    else:
        X_dim = Y_dim = 1
    
    # training alignment
    l_align_train = np.median(np.diag(X_rep.dot(Y_rep.T)))
    
    # test set alignment
    l_align_test = np.median(np.diag(X_rep_test.dot(Y_rep_test.T)))
    
    # downstream
    topk_acc_train = 1 - top_k_err(torch.Tensor(X_rep), torch.Tensor(Y_rep), np.int64(K_frac * n))
    topk_acc_test = 1 - top_k_err(torch.Tensor(X_rep_test), torch.Tensor(Y_rep_test), np.int64(K_frac * n_test))
    acc1_train, acc2_train = match_acc(torch.Tensor(X_rep), torch.Tensor(Y_rep), lab1_train, lab2_train)
    acc1_test, acc2_test = match_acc(torch.Tensor(X_rep_test), torch.Tensor(Y_rep_test), lab1_test, lab2_test)
    acc1_tr, acc1_te, acc2_tr, acc2_te, acc_tr, acc_te = np.mean(acc1_train), np.mean(acc1_test), np.mean(acc2_train), np.mean(acc2_test), np.mean(acc1_train * acc2_train), np.mean(acc1_test * acc2_test)
    
    return X_dim, Y_dim, l_align_train, l_align_test, topk_acc_train, topk_acc_test, acc1_tr, acc1_te, acc2_tr, acc2_te, acc_tr, acc_te, tau_seq, loss_clip, loss_align


def reload(i_rp, root_dir, dataset, joint,n,arch,link,outdim,K_frac,repN):
    
    if dataset == 'synthetic':
        d_x, d_y, d_z = 20, 20, 5
    elif dataset[:4] == 'yfcc' or dataset[:4] == 'imag':
        d_x, d_y, d_z = 1024, 768, 0
        if joint == False:
            d_x = 768
    elif dataset == 'citeseq':
        d_x, d_y, d_z = 200, 24, 0
            
    # print(f"=========================\noutdim = {outdim}")
    
    sub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}'
    subsub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}/file_{d_z}_{arch}_{link}_{repN}_{outdim}'
    
    file_dir = f'{subsub_root}/rep_{i_rp}_{n}_{d_x}_{d_y}_{d_z}_{dataset}_{outdim}_{joint}.npz'
    res = np.load(file_dir, allow_pickle=True)
    
    X_rep=res['X_rep']
    Y_rep=res['Y_rep']
    X_rep_test=res['X_rep_test']
    Y_rep_test=res['Y_rep_test']
    n_test = X_rep_test.shape[0]
    tau_seq=res['tau_seq']
    loss_clip=res['loss_clip']
    loss_align=res['loss_align']

    file_dir = f'{subsub_root}/acc_{i_rp}_{n}_{d_x}_{d_y}_{d_z}_{dataset}_{outdim}_{joint}.npz'
    res_acc = np.load(file_dir, allow_pickle=True)

    if dataset != "synthetic":
        topk_acc_train = res_acc['topk_acc_train']
        topk_acc_test = res_acc['topk_acc_test']
    else:
        K_fc = 1/n_test
        topk_acc_train = 1 - top_k_err(torch.Tensor(X_rep), torch.Tensor(Y_rep), np.int64(K_fc*n))
        topk_acc_test = 1 - top_k_err(torch.Tensor(X_rep_test), torch.Tensor(Y_rep_test), np.int64(K_fc*n_test))
    
    acc1_train = res_acc['acc1_train'] 
    acc2_train = res_acc['acc2_train'] 
    acc1_test = res_acc['acc1_test'] 
    acc2_test = res_acc['acc2_test'] 

    n, n_test = X_rep.shape[0], X_rep_test.shape[0]

    if outdim > 1:
        mle = MLE()
        X_dim = mle.fit_transform(X_rep_test)
        Y_dim = mle.fit_transform(Y_rep_test)
    else:
        X_dim = Y_dim = 1
    
    # training alignment
    l_align_train = np.median(np.diag(X_rep.dot(Y_rep.T)))
    
    # test set alignment
    l_align_test = np.median(np.diag(X_rep_test.dot(Y_rep_test.T)))
    
    # downstream
    acc1_tr, acc1_te, acc2_tr, acc2_te, acc_tr, acc_te = np.mean(acc1_train), np.mean(acc1_test), np.mean(acc2_train), np.mean(acc2_test), np.mean(acc1_train * acc2_train), np.mean(acc1_test * acc2_test)
    
    return X_dim, Y_dim, l_align_train, l_align_test, topk_acc_train, topk_acc_test, acc1_tr, acc1_te, acc2_tr, acc2_te, acc_tr, acc_te, tau_seq, loss_clip, loss_align



def topk_acc(i_rp, root_dir, dataset,joint,n,arch,link,outdim,K_frac):
    
    if dataset == 'synthetic':
        d_x, d_y, d_z = 20, 20, 10
    elif dataset[:4] == 'yfcc' or dataset[:4] == 'imag':
        d_x, d_y, d_z = 1024, 768, 0
        if joint == False:
            d_x = 768
    elif dataset == 'citeseq':
        d_x, d_y, d_z = 200, 24, 0
    
    sub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}'
    subsub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}/file_{d_z}_{arch}_{link}_{repN}_{outdim}'
    
    file_dir = f'{subsub_root}/rep_{i_rp}_{n}_{d_x}_{d_y}_{d_z}_{dataset}_{outdim}_{joint}.npz'
    res = np.load(file_dir, allow_pickle=True)
    
    X_rep=res['X_rep']
    Y_rep=res['Y_rep']
    X_rep_test=res['X_rep_test']
    Y_rep_test=res['Y_rep_test']
    n_test = X_rep_test.shape[0]
    tau_seq=res['tau_seq']

    n, n_test = X_rep.shape[0], X_rep_test.shape[0]
    
    # downstream
    topk_acc_train = 1 - top_k_err(torch.Tensor(X_rep), torch.Tensor(Y_rep), np.int64(K_frac * n))
    topk_acc_test = 1 - top_k_err(torch.Tensor(X_rep_test), torch.Tensor(Y_rep_test), np.int64(K_frac * n))

    return topk_acc_train, topk_acc_test