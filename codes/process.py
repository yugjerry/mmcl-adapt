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
from process_utils import *

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



print("\nParsing input parameters...")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='synthetic')
parser.add_argument('--arch', type=str, default='deep')
parser.add_argument('--link', type=str, default='linear')
parser.add_argument('--repN', type=int, default=50)
parser.add_argument('--joint', type=str, default=False)
args = parser.parse_args()



arch = args.arch
link = args.link
dataset = args.dataset
repN = args.repN
joint = args.joint


# parameters
root_dir = ''
data_dir = f'{root_dir}/data'


n = 10000

# dim_seq = np.int64(np.linspace(1,29,15))
dim_seq = np.int64(np.linspace(1,44,15))

if dataset == 'synthetic':
    d_x, d_y, d_z = 20, 20, 5
    n = 10000
elif dataset == 'yfcc':
    # joint = True
    d_x, d_y, d_z = 1024, 768, 0
    dataset_nam = 'yfcc'
    M = 10000
    n = 8000
    dir1 = f'{data_dir}/yfcc/%s_image_features_%s_%s.npy'%(dataset_nam,joint,M)
    dir2 = f'{data_dir}/yfcc/%s_label_features_%s_%s.npy'%(dataset_nam,joint,M)
    image_features = np.load(dir1, mmap_mode='r')
    text_features = np.load(dir2, mmap_mode='r')

    X = image_features
    Y = text_features
elif dataset == 'imagenetv2':
    # joint = False
    d_x, d_y, d_z = 1024, 768, 0
    if not joint:
        d_x = 768
        dataset = dataset+"_sep"
    dataset_nam = 'imagenetv2'
    M = 10000
    n = 8000
    dir1 = f'{data_dir}/imagenetv2/%s_image_embeds_%s_%s.npy'%(dataset_nam,joint,M)
    dir2 = f'{data_dir}/imagenetv2/%s_label_embeds_%s_%s.npy'%(dataset_nam,joint,M)
    image_features = np.load(dir1, mmap_mode='r')
    text_features = np.load(dir2, mmap_mode='r')

    X = image_features
    Y = text_features
elif dataset == 'citeseq':
    d_x, d_y, d_z = 200, 24, 0

    n = 10000

    dir1 = f'{data_dir}/citeseq/rna_pca.csv'
    dir2 = f'{data_dir}/citeseq/adt_pca.csv'
    lab1_dir = f'{data_dir}/citeseq/lab1.csv'
    lab2_dir = f'{data_dir}/citeseq/lab2.csv'
    df1 = pd.read_csv(dir1)
    X = np.array(df1.drop(df1.columns[0], axis=1))
    df2 = pd.read_csv(dir2)
    Y = np.array(df2.drop(df2.columns[0], axis=1))




l_train_seq = []
l_test_seq = []
topk_acc_train_seq = []
topk_acc_test_seq = []
acc1_train_seq = []
acc1_test_seq = []
acc2_train_seq = []
acc2_test_seq = []
acc_train_seq = []
acc_test_seq = []

topk_acc_train_std = []
topk_acc_test_std = []
acc1_train_std = []
acc1_test_std = []
acc2_train_std = []
acc2_test_std = []
acc_train_std = []
acc_test_std = []

X_dim_seq = []
Y_dim_seq = []
X_dim_std = []
Y_dim_std = []

tau_seq_seq = []
loss_seq = []
loss_align_seq = []

K_frac = 5e-3

print('processing...\n')

for outdim in dim_seq:
    
    l_align_train = np.zeros(repN)
    l_align_test = np.zeros(repN)
    
    topk_acc_train = np.zeros(repN)
    topk_acc_test = np.zeros(repN)
    acc1_tr = np.zeros(repN)
    acc1_te = np.zeros(repN)
    acc2_tr = np.zeros(repN)
    acc2_te = np.zeros(repN)
    acc_tr = np.zeros(repN)
    acc_te = np.zeros(repN)
#     loss_clip = np.zeros(repN)
    loss_align_ave = np.zeros(repN)

    x_dim = np.zeros(repN)
    y_dim = np.zeros(repN)
    
    for i_rp in tqdm(range(repN)):
        x_dim[i_rp], y_dim[i_rp], l_align_train[i_rp], l_align_test[i_rp], topk_acc_train[i_rp], topk_acc_test[i_rp], acc1_tr[i_rp], acc1_te[i_rp], acc2_tr[i_rp], acc2_te[i_rp], acc_tr[i_rp], acc_te[i_rp], tau_seq, loss_clip, loss_align = postprocess(i_rp,root_dir,dataset,joint,n,arch,link,outdim,K_frac,repN)

        loss_align_ave[i_rp] = -np.min(loss_align[-100:])
    
    X_dim_seq.append(np.nanmean(x_dim))
    Y_dim_seq.append(np.nanmean(y_dim))
    X_dim_std.append(np.nanstd(x_dim))
    Y_dim_std.append(np.nanstd(y_dim))

    l_train_seq.append(np.mean(l_align_train))
    l_test_seq.append(np.mean(l_align_test))
    
    topk_acc_train_seq.append(np.mean(topk_acc_train))
    topk_acc_test_seq.append(np.mean(topk_acc_test))
    acc1_train_seq.append(np.median(acc1_tr))
    acc1_test_seq.append(np.mean(acc1_te))
    acc2_train_seq.append(np.mean(acc2_tr))
    acc2_test_seq.append(np.mean(acc2_te))
    acc_train_seq.append(np.mean(acc_tr))
    acc_test_seq.append(np.mean(acc_te))
    
    topk_acc_train_std.append(np.std(topk_acc_train))
    topk_acc_test_std.append(np.std(topk_acc_test))
    acc1_train_std.append(np.std(acc1_tr))
    acc1_test_std.append(np.std(acc1_te))
    acc2_train_std.append(np.std(acc2_tr))
    acc2_test_std.append(np.std(acc2_te))
    acc_train_std.append(np.std(acc_tr))
    acc_test_std.append(np.std(acc_te))
    
    loss_seq.append(np.mean(loss_clip))
    loss_align_seq.append(np.mean(loss_align_ave))
                          
    tau_seq_seq.append(tau_seq)


sub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}'

file_dir = f'{sub_root}/acc_{n}_{d_x}_{d_y}_{d_z}_{dataset}_{dim_seq[-1]}_{joint}_{repN}.npz'
print(file_dir)

np.savez(file_dir, topk_acc_train=topk_acc_train_seq, topk_acc_test=topk_acc_test_seq, topk_acc_train_std=topk_acc_train_std, topk_acc_test_std=topk_acc_test_std,
    acc1_train=acc1_train_seq, acc1_test=acc1_test_seq, acc1_train_std=acc1_train_std, acc1_test_std=acc1_test_std,
    acc2_train=acc2_train_seq, acc2_test=acc2_test_seq, acc2_train_std=acc2_train_std, acc2_test_std=acc2_test_std,
    X_dim=X_dim_seq, Y_dim=Y_dim_seq, X_dim_std=X_dim_std, Y_dim_std=Y_dim_std
    )

res = np.load(file_dir, allow_pickle=True)
topk_acc_train_seq = res['topk_acc_train']
topk_acc_test_seq = res['topk_acc_test']
acc1_train_seq = res['acc1_train']
acc1_test_seq = res['acc1_test']
acc2_train_seq = res['acc2_train']
acc2_test_seq = res['acc2_test']

topk_acc_train_std = res['topk_acc_train_std']
topk_acc_test_std = res['topk_acc_test_std']
acc1_train_std = res['acc1_train_std']
acc1_test_std = res['acc1_test_std']
acc2_train_std = res['acc2_train_std']
acc2_test_std = res['acc2_test_std']

X_dim_seq = res['X_dim']
Y_dim_seq = res['Y_dim']
X_dim_std = res['X_dim_std']
Y_dim_std = res['Y_dim_std']

topk_acc_train_seq = np.array(topk_acc_train_seq)
topk_acc_test_seq = np.array(topk_acc_test_seq)
acc1_train_seq = np.array(acc1_train_seq)
acc1_test_seq = np.array(acc1_test_seq)
acc2_train_seq = np.array(acc2_train_seq)
acc2_test_seq = np.array(acc2_test_seq)
acc_train_seq = np.array(acc_train_seq)
acc_test_seq = np.array(acc_test_seq)

topk_acc_train_std = np.array(topk_acc_train_std)
topk_acc_test_std = np.array(topk_acc_test_std)
acc1_train_std = np.array(acc1_train_std)
acc1_test_std = np.array(acc1_test_std)
acc2_train_std = np.array(acc2_train_std)
acc2_test_std = np.array(acc2_test_std)
acc_train_std = np.array(acc_train_std)
acc_test_std = np.array(acc_test_std)

X_dim_seq = np.array(X_dim_seq)
Y_dim_seq = np.array(Y_dim_seq)
X_dim_std = np.array(X_dim_std)
Y_dim_std = np.array(Y_dim_std)

def plot_errbar(axs, x_seq, data_seq, data_std, label, color, ecolor):
    axs.plot(x_seq, data_seq, color=color, marker='o', linestyle='--', label=label)
    axs.fill_between(x_seq, data_seq-data_std, data_seq+data_std, alpha=0.5, 
                        edgecolor=ecolor, facecolor=ecolor)
    return 0


sub_root = f'{root_dir}/{dataset}/file_{d_z}_{arch}_{link}_{repN}'
plot_dir = f'{sub_root}/plot_dim_{dataset}_{joint}_{repN}.pdf'

if dataset == 'citeseq':
    fig,ax = plt.subplots(1,4,figsize=(18,4))


    plot_errbar(ax[0], dim_seq, topk_acc_train_seq, topk_acc_train_std, 'in-sample', 'blue', 'lightblue')
    plot_errbar(ax[0], dim_seq, topk_acc_test_seq, topk_acc_test_std, 'out-of-sample', 'red', 'lightpink')
    ax[0].legend(loc='best')
    ax[0].set_ylim((0,1.1))
    ax[0].set_xlabel('output dimension')
    ax[0].set_title(f'top-{K_frac*100}% accuracy')


    plot_errbar(ax[1], dim_seq, acc1_train_seq, acc1_train_std, 'in-sample', 'blue', 'lightblue')
    plot_errbar(ax[1], dim_seq, acc1_test_seq, acc1_test_std, 'out-of-sample', 'red', 'lightpink')
    ax[1].set_ylim((0,1.1))
    ax[1].legend(loc='best')
    ax[1].set_xlabel('output dimension')
    ax[1].set_title('cell type accuracy (1)')


    plot_errbar(ax[2], dim_seq, acc2_train_seq, acc2_train_std, 'in-sample', 'blue', 'lightblue')
    plot_errbar(ax[2], dim_seq, acc2_test_seq, acc2_test_std, 'out-of-sample', 'red', 'lightpink')
    ax[2].set_ylim((0,1.1))
    ax[2].legend(loc='best')
    ax[2].set_xlabel('output dimension')
    ax[2].set_title('cell type accuracy (2)')

    plot_errbar(ax[3], dim_seq[1:], X_dim_seq[1:], X_dim_std[1:], 'f(X)', 'darkgreen', 'lightgreen')
    plot_errbar(ax[3], dim_seq[1:], Y_dim_seq[1:], Y_dim_std[1:], 'g(Y)', 'orange', 'orange')
    ax[3].legend(loc='best')
    ax[3].set_xlabel('output dimension')
    ax[3].set_title('MLE-intrinsic dimension')
    ax[3].set_ylim((0,15))

elif dataset != "synthetic":
    fig,ax = plt.subplots(1,3,figsize=(15,4))
    plot_errbar(ax[0], dim_seq, topk_acc_train_seq, topk_acc_train_std, 'in-sample', 'blue', 'lightblue')
    plot_errbar(ax[0], dim_seq, topk_acc_test_seq, topk_acc_test_std, 'out-of-sample', 'red', 'lightpink')
    ax[0].set_ylim((0,1.1))
    ax[0].legend(loc='best')
    ax[0].set_xlabel('output dimension')
    ax[0].set_title(f'top-{K_frac*100}% accuracy')


    plot_errbar(ax[1], dim_seq, acc1_train_seq, acc1_train_std, 'in-sample', 'blue', 'lightblue')
    plot_errbar(ax[1], dim_seq, acc1_test_seq, acc1_test_std, 'out-of-sample', 'red', 'lightpink')
    ax[1].set_ylim((0,1.1))
    ax[1].legend(loc='best')
    ax[1].set_xlabel('output dimension')
    ax[1].set_title('classification accuracy')

    plot_errbar(ax[2], dim_seq[1:], X_dim_seq[1:], X_dim_std[1:], 'f(X)', 'darkgreen', 'lightgreen')
    plot_errbar(ax[2], dim_seq[1:], Y_dim_seq[1:], Y_dim_std[1:], 'g(Y)', 'orange', 'orange')
    ax[2].legend(loc='best')
    ax[2].set_xlabel('output dimension')
    ax[2].set_title('MLE-intrinsic dimension')
    ax[2].set_ylim((0,10))

else:
    fig,ax = plt.subplots(1,2,figsize=(9,4))
    plot_errbar(ax[0], dim_seq, topk_acc_train_seq, topk_acc_train_std, 'in-sample', 'blue', 'lightblue')
    plot_errbar(ax[0], dim_seq, topk_acc_test_seq, topk_acc_test_std, 'out-of-sample', 'red', 'lightpink')
    ax[0].set_ylim((0,1.1))
    ax[0].legend(loc='best')
    ax[0].set_xlabel('output dimension')
    ax[0].set_title(f'top-{100/2000}% accuracy')

    plot_errbar(ax[1], dim_seq[1:], X_dim_seq[1:], X_dim_std[1:], 'f(X)', 'darkgreen', 'lightgreen')
    plot_errbar(ax[1], dim_seq[1:], Y_dim_seq[1:], Y_dim_std[1:], 'g(Y)', 'orange', 'orange')
    ax[1].legend(loc='best')
    ax[1].set_xlabel('output dimension')
    ax[1].set_title('MLE-intrinsic dimension')
    ax[1].set_ylim((0,20))

plt.savefig(plot_dir, bbox_inches='tight')
