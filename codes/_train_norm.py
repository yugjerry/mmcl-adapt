import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import sys
from matplotlib import pyplot as plt
import warnings
from scipy.linalg import qr
import seaborn as sns
from tqdm import tqdm
from pytorch_metric_learning import losses
from sklearn.decomposition import PCA
import skdim
import argparse
import math

from utils import *



sns.set_theme()
sns.set_context("notebook")


def train_once(i_rp, root_dir, repN, dataset, n, n_append, n_test, d_x, d_y, d_z, arch, middim, outdim, tau_lower, K_frac, joint, link, lr, wd, batch_size, max_epochs):
    if dataset == 'yfcc':
        X, Y, X_test, Y_test, lab1_train, lab1_test = data_gen_yfcc(f'{root_dir}/data', n + n_append, n_test, d_x, d_z, d_y, joint=joint, link=link)
        lab2_train, lab2_test = np.zeros(n), np.zeros(n_test)
        d_x, d_y = X.shape[1], Y.shape[1]
        d_z = 0
        n_test = X_test.shape[0]
        N = n + n_test
        middim = max(d_x, d_y)
        if joint == False:
            dataset = "yfcc_sep"
    elif dataset == 'imagenetv2':
        X, Y, X_test, Y_test, lab1_train, lab1_test = data_gen_yfcc(f'{root_dir}/data', n + n_append, n_test, d_x, d_z, d_y, joint=joint, dataset_nam=dataset, link=link)
        lab2_train, lab2_test = np.zeros(n), np.zeros(n_test)
        
        ## add noise to avoid singularity
        Y +=  np.random.normal(0,0.1,Y.shape)
        Y_test +=  np.random.normal(0,0.1,Y_test.shape)

        d_x, d_y = X.shape[1], Y.shape[1]
        d_z = 0
        # n, n_test = X.shape[0], X_test.shape[0]
        n_test = X_test.shape[0]
        N = n + n_test
        middim = max(d_x, d_y)
        if joint == False:
            dataset = "imagenetv2_sep"
    elif dataset == 'synthetic':
        X, Y, X_test, Y_test = data_gen_syn(n + n_append, n_test, d_x, d_z, d_y, link=link)
        lab2_train, lab2_test = np.zeros(n), np.zeros(n_test)
        lab1_train, lab1_test = np.zeros(n), np.zeros(n_test)    
    elif dataset == 'citeseq':
        is_pca = True
        X, Y, X_test, Y_test, lab1_train, lab1_test, lab2_train, lab2_test = data_gen_citeseq(f'{root_dir}/data', n + n_append, n_test, d_x, d_z, d_y, joint=joint, pca=is_pca, link=link)
        d_x, d_y = X.shape[1], Y.shape[1]
        d_z = 0
        n_test = X_test.shape[0]
        N = n + n_test
        middim = max(d_x, d_y)

    # splitting
    X_append = X[n:,:]
    Y_append = Y[n:,:]
    X = X[:n,:]
    Y = Y[:n,:]

    print([X.shape, X_test.shape, X_append.shape])
    
    norm = False

    if arch == 'linear':
        model_x = LinearNet(d_x, middim, outdim, tau_lower, norm).to(device)
        model_y = LinearNet(d_y, middim, outdim, tau_lower, norm).to(device)
    elif arch == 'nonlinear':
        model_x = NonLinearNet(d_x, middim, outdim, tau_lower, norm).to(device)
        model_y = NonLinearNet(d_y, middim, outdim, tau_lower, norm).to(device)
    else:
        model_x = NonLinearNetD(d_x, 50, outdim, tau_lower, norm).to(device)
        model_y = NonLinearNetD(d_y, 50, outdim, tau_lower, norm).to(device)


    optimizer_x = optim.Adam(model_x.parameters(), lr=lr, weight_decay=wd)
    optimizer_y = optim.Adam(model_y.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss().to(device)
    smooth_loss = 0
    loss_clip = []
    loss_align = []
    loss_x = []
    loss_y = []
    tau_seq = []


    for epoch in range(max_epochs):
        X, Y = torch.Tensor(X).to(device), torch.Tensor(Y).to(device)
        X_append, Y_append = torch.Tensor(X_append).to(device), torch.Tensor(Y_append).to(device)

        losses_clip = []
        losses_align = []
        losses_x = []
        losses_y = []

        for batch_idx in range(n // batch_size):
            optimizer_x.zero_grad()
            optimizer_y.zero_grad()
            indices = range(batch_idx*batch_size, batch_idx*batch_size+batch_size)
            x, y = X[indices,:].to(device), Y[indices,:].to(device)
            h_x = model_x(x, X_append).to(device)
            h_y = model_y(y, Y_append).to(device)

            if tau_tune:
                if dataset == 'synthetic':
                    tau_lr_frac = 4
                else:
                    tau_lr_frac = 2
                tau = (tau_lr_frac*model_x.logit_scale).exp() + tau_lower
                tau_seq.append(tau.detach().numpy())
            else:
                tau = tau_fix
            
            # Calculating
            logits = (h_x @ h_y.T)
            logits = logits/tau
            bn = logits.shape[1]      # number of samples
            labels = torch.arange(bn) # Create labels tensor

            images_similarity = h_x @ h_x.T
            texts_similarity = h_y @ h_y.T
            loss_x_ = F.cross_entropy(logits, labels, reduction='mean')
            loss_y_ = F.cross_entropy(logits.T, labels, reduction='mean')

            loss = (loss_x_ + loss_y_) / 2.0
            
            # decomposition of loss
            losses_clip.append(loss.item())
            h_x_unit = F.normalize(h_x)
            h_y_unit = F.normalize(h_y)
            losses_align.append(-torch.mean(torch.diagonal(h_x_unit @ h_y_unit.T)).detach().numpy())
            l_align = -torch.mean(torch.diagonal(h_x @ h_y.T)).detach().numpy()
            losses_x.append(loss_x_.item()/2.0 - l_align)
            losses_y.append(loss_y_.item()/2.0 - l_align)


            smooth_loss = 0.2 * loss.item() + 0.8 * smooth_loss
            loss.backward()
            optimizer_x.step()
            optimizer_y.step()

        loss_clip.append(np.mean(losses_clip))
        loss_align.append(np.mean(losses_align))
        loss_x.append(np.mean(losses_x))
        loss_y.append(np.mean(losses_y))


    sub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}'
    subsub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}/file_{d_z}_{arch}_{link}_{repN}_{outdim}'
    if not os.path.exists(subsub_root):
        os.makedirs(subsub_root)


    X_test, Y_test = torch.Tensor(X_test), torch.Tensor(Y_test)
    
    X_rep = model_x(X, X_append).detach().numpy()
    Y_rep = model_y(Y, Y_append).detach().numpy()
    X_rep_test = model_x(X_test, X_append).detach().numpy()
    Y_rep_test = model_y(Y_test, Y_append).detach().numpy()

    file_dir = f'{subsub_root}/rep_{i_rp}_{n}_{d_x}_{d_y}_{d_z}_{dataset}_{outdim}_{joint}.npz'
    np.savez(file_dir, X_rep=X_rep, Y_rep=Y_rep, X_rep_test=X_rep_test, Y_rep_test=Y_rep_test, 
        lab1_train=lab1_train, lab1_test=lab1_test, lab2_train=lab2_train, lab2_test=lab2_test,
        tau_seq=np.array(tau_seq), 
        loss_clip=loss_clip, loss_align=loss_align)


    # postprocess

    # training alignment
    l_align_train = np.median(np.diag(X_rep.dot(Y_rep.T)))
    
    # test set alignment
    l_align_test = np.median(np.diag(X_rep_test.dot(Y_rep_test.T)))
    
    # downstream
    topk_acc_train = 1 - top_k_err(torch.Tensor(X_rep), torch.Tensor(Y_rep), np.int64(n*K_frac))
    topk_acc_test = 1 - top_k_err(torch.Tensor(X_rep_test), torch.Tensor(Y_rep_test), np.int64(n_test*K_frac))
    acc1_train, acc2_train = match_acc(torch.Tensor(X_rep), torch.Tensor(Y_rep), lab1_train, lab2_train)
    acc1_test, acc2_test = match_acc(torch.Tensor(X_rep_test), torch.Tensor(Y_rep_test), lab1_test, lab2_test)
    acc1_tr, acc1_te, acc2_tr, acc2_te, acc_tr, acc_te = np.mean(acc1_train), np.mean(acc1_test), np.mean(acc2_train), np.mean(acc2_test), np.mean(acc1_train * acc2_train), np.mean(acc1_test * acc2_test)
    
    file_dir = f'{subsub_root}/acc_{i_rp}_{n}_{d_x}_{d_y}_{d_z}_{dataset}_{outdim}_{joint}.npz'
    np.savez(file_dir, topk_acc_train=topk_acc_train, topk_acc_test=topk_acc_test,
        acc1_train=acc1_train, acc1_test=acc1_test,
        acc2_train=acc2_train, acc2_test=acc2_test)


    return 0





wd_name = os.getcwd()
seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)

device = "cpu" if not torch.cuda.is_available() else torch.device("cuda")
print("Device is :: {}...".format(device))


print("\nParsing input parameters...")
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=10000)
parser.add_argument('--n_append', type=int, default=1000)
parser.add_argument('--dataset', type=str, default='synthetic')
parser.add_argument('--joint', type=bool, default=False)
parser.add_argument('--middim', type=int, default=50)
parser.add_argument('--outdim', type=int, default=20)
parser.add_argument('--d_x', type=int, default=20)
parser.add_argument('--d_y', type=int, default=20)
parser.add_argument('--d_z', type=int, default=10)
parser.add_argument('--tau_tune', type=bool, default=True)
parser.add_argument('--epoch_num', type=int, default=800)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--arch', type=str, default='deep')
parser.add_argument('--link', type=str, default='linear')
parser.add_argument('--repN', type=int, default=2)
args = parser.parse_args()

root_dir = ''

# parameters
repN = args.repN
n = args.n
n_append = args.n_append
n_test = 2000

# data generation
print("\nGenerating data...")
dataset = args.dataset
joint = args.joint
link = args.link

d_x, d_y, d_z = args.d_x, args.d_y, args.d_z
outdim = args.outdim
dim_seq = np.int64(np.linspace(1,44,15))
outdim = dim_seq[outdim-1]

if args.dataset != 'synthetic':
    d_z = 0 
if dataset == 'yfcc':
    joint = True
    if n >= 10000:
        print("0 test sample size! STOP!")


# model architecture
arch = args.arch
middim = args.middim
K_frac = 5e-3
tau_lower = 1e-4

tau_tune = args.tau_tune
max_epochs = args.epoch_num
batch_size = args.batch_size
lr = args.lr
wd = args.wd


print([dataset, link, joint])



# training

if dataset == "synthetic":
    link_seq = ['linear', 'nonlinear']
else:
    link_seq = [link]

for link in link_seq:
    print("\nStarting training...")

    print(f"Output dim = {outdim}")

    sub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}'
    subsub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}/file_{d_z}_{arch}_{link}_{repN}_{outdim}'

    if not os.path.exists(subsub_root):
        os.makedirs(subsub_root)

    for i_rp in tqdm(range(repN)):

        train_once(i_rp, root_dir, repN, dataset, n, n_append, n_test, d_x, d_y, d_z, arch, middim, outdim, tau_lower, K_frac, joint, link, lr, wd, batch_size, max_epochs)

            