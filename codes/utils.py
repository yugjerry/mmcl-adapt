import os
import numpy as np
import sys
from matplotlib import pyplot as plt
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.linalg import qr, sqrtm
from scipy.linalg import sqrtm
import seaborn as sns
from tqdm import tqdm
from pytorch_metric_learning import losses
import skdim
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def id_est(data):
    lpca_ = skdim.id.lPCA('FO',alphaFO=0.1).fit(data)
    lpca = skdim.id.lPCA('FO',alphaFO=0.1).fit_pw(data,
                                  n_neighbors = 100,
                                  n_jobs = 1)
    return lpca_.dimension_, np.mean(lpca.dimension_pw_)

def rescale_by_max_absolute(tensor):

    max_abs = torch.max(torch.abs(tensor),dim=0)[0]
    return tensor / max_abs

def rescale_by_ave_norm(tensor):

    v_norm2 = torch.linalg.vector_norm(tensor, ord=2, dim=1)**2
    v_norm2_loo = - v_norm2 + torch.sum(v_norm2)
    norms_mean = torch.sqrt(torch.sum(v_norm2_loo)/(tensor.shape[0]-1))
    # norms_mean = torch.norm(self.linear2.weight, 'fro')
    return tensor / norms_mean

def info_nce_loss(features, temperature, batch_size, num_aug): #  code source from SimCLR paper: https://github.com/sthalles/SimCLR


    labels = torch.cat([torch.arange(batch_size) for i in range(num_aug)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    mask1 = torch.ones((labels.shape[0]//2, labels.shape[0]//2))
    mask = torch.block_diag(mask1, mask1).bool()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives / temperature], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    logits = logits / temperature

    return logits, labels

relu = nn.ReLU()
tanh = nn.Tanh()

tau_init = 1e0
tau_lower = 1e-5

class NonLinearNet(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, tau_lower, norm):
        super(NonLinearNet, self).__init__()
        self.input_dim = input_dim
        self.middle_dim = middle_dim
        self.output_dim = output_dim
        self.tau_lower = tau_lower
        self.linear1 = nn.Linear(input_dim, middle_dim, bias=False)
        self.linear2 = nn.Linear(middle_dim, output_dim, bias=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(tau_init))
        self.norm = norm

    def rec_gaus_norm(self):
        N = 10000
        d_x = self.input_dim
        x_ = torch.Tensor(np.random.multivariate_normal(mean=np.zeros(d_x),cov=np.eye(d_x),size=N).reshape((N,d_x)))
        z_ = self.linear1(x_)
        z_ = relu(z_)
        z_ = self.linear2(z_)
        z_norm2 = torch.linalg.norm(z_, ord=2, dim=1)**2
        return torch.sqrt(torch.mean(z_norm2))

    def _rescale_by_ave_norm(self, tensor):
        norms_mean = self.rec_gaus_norm()
        return tensor / norms_mean
        
    def forward(self, x, x_append):
        z = self.linear1(x)
        z = relu(z)
        z = self.linear2(z)
        # z = F.normalize(z)

        z_append = self.linear1(x_append)
        z_append = relu(z_append)
        z_append = self.linear2(z_append)
        z_norm2 = torch.linalg.norm(z_append, ord=2, dim=1)

        if self.norm:
            z = F.normalize(z)
        else:
            z = z / torch.mean(z_norm2)
        
        tau = self.logit_scale.exp() + self.tau_lower
        
        return z


class NonLinearNetD(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, tau_lower, norm):
        super(NonLinearNetD, self).__init__()
        self.input_dim = input_dim
        self.middle_dim = middle_dim
        self.output_dim = output_dim
        self.tau_lower = tau_lower
        self.norm = norm
        self.linear1 = nn.Linear(input_dim, middle_dim, bias=True)
        self.linear2 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear3 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear4 = nn.Linear(middle_dim, output_dim, bias=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(tau_init))
        
    def forward(self, x, x_append):
        z = self.linear1(x)
        z = relu(z)
        z = self.linear2(z)
        z = relu(z)
        z = self.linear3(z)
        z = relu(z)
        z = self.linear4(z)

        z_append = self.linear1(x_append)
        z_append = relu(z_append)
        z_append = self.linear2(z_append)
        z_append = relu(z_append)
        z_append = self.linear3(z_append)
        z_append = relu(z_append)
        z_append = self.linear4(z_append)

        z_norm2 = torch.linalg.norm(z_append, ord=2, dim=1)
        
        if self.norm:
            z = F.normalize(z)
        else:
            z = z / torch.mean(z_norm2)

        
        tau = self.logit_scale.exp() + self.tau_lower
        
        return z

class DeepNet(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, tau_lower):
        super(DeepNet, self).__init__()
        self.input_dim = input_dim
        self.middle_dim = middle_dim
        self.output_dim = output_dim
        self.tau_lower = tau_lower
        self.linear1 = nn.Linear(input_dim, middle_dim, bias=True)
        self.linear2 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear3 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear4 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear5 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear6 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear7 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear8 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear9 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear10 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear11 = nn.Linear(middle_dim, middle_dim, bias=True)
        self.linear12 = nn.Linear(middle_dim, output_dim, bias=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(tau_init))
        
    def forward(self, x):
        z = self.linear1(x)
        z = relu(z)
        z = self.linear2(z)
        z = relu(z)
        z = self.linear3(z)
        z = relu(z)
        z = self.linear4(z)
        z = relu(z)
        z = self.linear5(z)
        z = relu(z)
        z = self.linear6(z)
        z = relu(z)
        z = self.linear7(z)
        z = relu(z)
        z = self.linear8(z)
        z = relu(z)
        z = self.linear9(z)
        z = relu(z)
        z = self.linear10(z)
        z = relu(z)
        z = self.linear11(z)
        z = relu(z)
        z = self.linear12(z)
        z = F.normalize(z)
        
        tau = self.logit_scale.exp() + self.tau_lower
        
        return z
        return self.layers(x)



class LinearNet(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, tau_lower, norm, M):
        super(LinearNet, self).__init__()
        self.input_dim = input_dim
        self.middle_dim = middle_dim
        self.output_dim = output_dim
        self.tau_lower = tau_lower
        self.linear1 = nn.Linear(input_dim, middle_dim, bias=False)
        self.linear2 = nn.Linear(middle_dim, output_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(tau_init))
        self.norm = norm
        self.M = M
    
    def _rescale_by_ave_norm(self, tensor):
        norms_mean = torch.norm(self.linear2.weight, 'fro')
        return tensor / norms_mean
    
    def forward(self, x, x_append):
        z = self.linear1(x)
        z = self.linear2(z)

        z_append = self.linear1(x_append)
        z_append = self.linear2(z_append)


        z_norm2 = torch.linalg.norm(z_append, ord=2, dim=1)
        # z_norm2 = torch.clamp(torch.linalg.norm(z_append, ord=2, dim=1), min=0.2)
        
        if self.norm:
            z = F.normalize(z)
        else:
            z = torch.clamp(z, max=10)
            z = z / torch.mean(z_norm2)
    
        tau = self.logit_scale.exp() + self.tau_lower
    
        return z

    
# nonlinear function that preserves gaussianity
def phi(x):
    n = len(x)
    res = 0
    if n == 2:
        res = np.sqrt(2)*x[0]*x[1]/np.sqrt(x[0]**2 + x[1]**2)
#         res = (x[0] * (x[0]>=0) - np.abs(x[1]) * (x[0]<0))
#     if n == 3:
#         res = (x[0]*x[1] + x[2])/np.sqrt(1 + x[0]**2)
    if n >= 3:
        for i in range(n-2):
            res += x[0]*x[i+1]/((1+x[0]**2)**((i+1)/2))
        res += x[n-1]/((1+x[0]**2)**((n-2)/2))
    return res

def gaus_Z(Y, d_z):
    n,d = Y.shape[0],Y.shape[1]
    res = np.zeros((n,d_z))
    for j in range(d_z):
        for i in range(n):
            x = np.concatenate((Y[i,j:d],Y[i,:j]))
            res[i,j] = phi(x)
    return res

def g1(M):
    resM = np.zeros((M.shape[0], M.shape[1]))
    for i in range(M.shape[0]):
        resM[i,:] = M[i,:] + M[i,0] * (M[i,0]>0)
        
    return resM

def g2(M):
    resM = np.zeros((M.shape[0], M.shape[1]))
    for i in range(M.shape[0]):
        resM[i,:] = M[i,:] + M[i,0]**2 - 2*M[i,1]*M[i,0]
        
    return resM


def gen(n,n_test,d_x,d_z,d_y,link):
    
    H = np.random.randn(d_x, d_x)
    Q, R = qr(H)
    A_x = Q[:,:d_z]
    H = np.random.randn(d_y, d_y)
    Q, R = qr(H)
    A_y = Q[:,:d_z]

    inv_A_x = np.linalg.inv(A_x.T.dot(A_x))
    P_A_x = A_x.dot(inv_A_x.dot(A_x.T))
    inv_A_y = np.linalg.inv(A_y.T.dot(A_y))
    P_A_y = A_y.dot(inv_A_y.dot(A_y.T))
    
    N = n + n_test
    if link == 'linear':
        X_res = np.random.multivariate_normal(mean=np.zeros(d_x),cov=np.eye(d_x),size=N).reshape((N,d_x))
        Y_res = np.random.multivariate_normal(mean=np.zeros(d_y),cov=np.eye(d_y),size=N).reshape((N,d_y))
        X_res_white = np.random.multivariate_normal(mean=np.zeros(d_x),cov=np.eye(d_x),size=n_test).reshape((n_test,d_x))
        Y_res_white = np.random.multivariate_normal(mean=np.zeros(d_y),cov=np.eye(d_y),size=n_test).reshape((n_test,d_y))

        Z_all = np.random.multivariate_normal(mean=np.zeros(d_z),cov=np.eye(d_z),size=N).reshape((N,d_z))
        X_all = Z_all.dot(A_x.T) + X_res.dot((np.eye(d_x) - P_A_x))
        Y_all = Z_all.dot(A_y.T) + Y_res.dot((np.eye(d_y) - P_A_y))
        Z_white = np.random.multivariate_normal(mean=np.zeros(d_z),cov=np.eye(d_z),size=n_test).reshape((n_test,d_z))
        X_white = Z_white.dot(A_x.T) + X_res_white.dot((np.eye(d_x) - P_A_x))
        Y_white = Z_white.dot(A_y.T) + Y_res_white.dot((np.eye(d_y) - P_A_y))
    if link == 'nonlinear':
        Y_all = np.random.multivariate_normal(mean=np.zeros(d_y),cov=np.eye(d_y),size=N).reshape((N,d_y))
        Z_all = gaus_Z(Y_all,d_z)
        U = np.random.multivariate_normal(mean=np.zeros(d_x),cov=np.eye(d_x),size=N).reshape((N,d_x))
        P = np.eye(d_x) - A_x.dot(A_x.T)
        X_all = Z_all.dot(A_x.T) + U.dot(P)
        Y_white = np.random.multivariate_normal(mean=np.zeros(d_y),cov=np.eye(d_y),size=n_test).reshape((n_test,d_y))
        Z_white = gaus_Z(Y_white,d_z)
        U = np.random.multivariate_normal(mean=np.zeros(d_x),cov=np.eye(d_x),size=n_test).reshape((n_test,d_x))
        X_white = Z_white.dot(A_x.T) + U.dot(P)
    if link == 'noise':
        Z_all = np.random.multivariate_normal(mean=np.zeros(d_z),cov=np.eye(d_z),size=N).reshape((N,d_z))
        X_all = np.random.multivariate_normal(mean=np.zeros(d_x),cov=np.eye(d_x),size=N).reshape((N,d_x))
        Y_all = np.random.multivariate_normal(mean=np.zeros(d_y),cov=np.eye(d_y),size=N).reshape((N,d_y))
        Z_white = np.random.multivariate_normal(mean=np.zeros(d_z),cov=np.eye(d_z),size=n_test).reshape((n_test,d_z))
        Y_white = np.random.multivariate_normal(mean=np.zeros(d_y),cov=np.eye(d_y),size=n_test).reshape((n_test,d_y))
        X_white = np.random.multivariate_normal(mean=np.zeros(d_x),cov=np.eye(d_x),size=n_test).reshape((n_test,d_x))
        
    A_x_ = np.hstack((A_x, np.zeros((d_x, d_x-d_z))))
    return X_all, Y_all, Z_all, X_white, Y_white, Z_white, A_x_
    

def gen_(n,n_test,d_x,d_z,d_y,link):
    
    H = np.random.randn(d_x, d_x)
    Q, R = qr(H)
    A_x = Q[:,:d_z]
    H = np.random.randn(d_y, d_y)
    Q, R = qr(H)
    A_y = Q[:,:d_z]

    noise = 0.0
    
    N = n + n_test

    X_res = np.random.multivariate_normal(mean=np.zeros(d_x),cov=np.eye(d_x),size=N).reshape((N,d_x))
        
    # generate Y: white noise
    Y_all = np.random.multivariate_normal(mean=np.zeros(d_y),cov=np.eye(d_y),size=N).reshape((N,d_y))
    
    # latent features for C1
    Z_all = Y_all[:,:d_z]

    if link == 'linear':
        
        # generate X from Z
        X_all_0 = X_res
        X_all_0[:,0] = (Z_all[:,0] + noise * X_res[:,0])/np.sqrt(1+noise**2)
        
        # higher dimension
        if d_z >= 2:
            X_all_0[:,1:d_z] = (Z_all[:,1:d_z] + noise * X_res[:,1:d_z])/np.sqrt(1+noise**2)
            
        X_all = X_all_0
    elif link == 'nonlinear':

        # generate X from Z
        X_all_0 = X_res
        X_all_0[:,0] = (Z_all[:,0] < 0) * Z_all[:,0]
        X_all_0[:,1] = (Z_all[:,1] + Z_all[:,0] > 0.5) * Z_all[:,1]

        # higher dimension
        if d_z > 2:
            for i in range(2,d_z):
                X_all_0[:,i] = (Z_all[:,i] > 0) * Z_all[:,i]
            
        X_all = X_all_0


    return X_all, Y_all, Z_all


def data_gen_citeseq(root_dir, n, n_test, d_x, d_z, d_y, joint, pca=True, link="linear"):

    if pca:
        dir1 = f'{root_dir}/citeseq/rna_pca.csv'
        dir2 = f'{root_dir}/citeseq/adt_pca.csv'
    else:
        dir1 = f'{root_dir}/citeseq/rna.csv'
        dir2 = f'{root_dir}/citeseq/adt.csv'
    df1 = pd.read_csv(dir1)
    X_all = np.array(df1.drop(df1.columns[0], axis=1))
    df2 = pd.read_csv(dir2)
    Y_all = np.array(df2.drop(df2.columns[0], axis=1))

    lab1_dir = f'{root_dir}/citeseq/lab1.csv'
    lab2_dir = f'{root_dir}/citeseq/lab2.csv'
    lab1 = pd.read_csv(lab1_dir)
    lab1 = np.array(lab1.drop(lab1.columns[0], axis=1)).ravel()
    lab2 = pd.read_csv(lab2_dir)
    lab2 = np.array(lab2.drop(lab2.columns[0], axis=1)).ravel()

    d_x = X_all.shape[1]
    d_y = Y_all.shape[1]
    d_z = 0

    idx = np.arange(X_all.shape[0])
    np.random.shuffle(idx)

    X_all = X_all[idx,:]
    Y_all = Y_all[idx,:]
    lab1 = lab1[idx]
    lab2 = lab2[idx]


    X = X_all[:n,:]
    Y = Y_all[:n,:]
    X_test = X_all[n:,:]
    Y_test = Y_all[n:,:]

    lab1_train = lab1[:n]
    lab2_train = lab2[:n]
    lab1_test = lab1[n:]
    lab2_test = lab2[n:]

    return X, Y, X_test, Y_test, lab1_train, lab1_test, lab2_train, lab2_test

def data_gen_yfcc(root_dir, n, n_test, d_x, d_z, d_y, joint, dataset_nam='yfcc', link="linear"):

    M = 10000

    dir_labs = f'{root_dir}/{dataset_nam}/%s_class_%s_%s.npy'%(dataset_nam,joint,M)
    class_labs = np.load(dir_labs, mmap_mode='r')

    # if joint:
    #     dir1 = f'{root_dir}/{dataset_nam}/%s_image_features_%s_%s.npy'%(dataset_nam,joint,M)
    #     dir2 = f'{root_dir}/{dataset_nam}/%s_label_features_%s_%s.npy'%(dataset_nam,joint,M)
    #     image_features = np.load(dir1, mmap_mode='r')
    #     text_features = np.load(dir2, mmap_mode='r')
        
    #     X_all = image_features
    #     Y_all = text_features

    #     # X_all = np.sqrt(image_features.shape[0])*image_features/np.linalg.norm(image_features, axis=0)
    #     # Y_all = np.sqrt(text_features.shape[0])*text_features/np.linalg.norm(text_features, axis=0)
    #     print('image features shape: (%s,%s)'%(X_all.shape[0], X_all.shape[1]))
    #     print('text features shape: (%s,%s)'%(Y_all.shape[0], Y_all.shape[1]))
    # else:
    dir1 = f'{root_dir}/{dataset_nam}/%s_image_embeds_%s_%s.npy'%(dataset_nam,joint,M)
    dir2 = f'{root_dir}/{dataset_nam}/%s_label_embeds_%s_%s.npy'%(dataset_nam,joint,M)
    image_features = np.load(dir1, mmap_mode='r')
    text_features = np.load(dir2, mmap_mode='r')

    X_all = image_features
    Y_all = text_features
    
    # X_all = np.sqrt(image_features.shape[0])*image_features/np.linalg.norm(image_features, axis=0)
    # Y_all = np.sqrt(text_features.shape[0])*text_features/np.linalg.norm(text_features, axis=0)
    print('image features shape: (%s,%s)'%(X_all.shape[0], X_all.shape[1]))
    print('text features shape: (%s,%s)'%(Y_all.shape[0], Y_all.shape[1]))
    
    d_x = X_all.shape[1]
    d_y = Y_all.shape[1]
    d_z = 0

    idx = np.arange(X_all.shape[0])
    np.random.shuffle(idx)

    X_all = X_all[idx,:]
    Y_all = Y_all[idx,:]
    class_labs = class_labs[idx]

        
    X = X_all[:n,:]
    Y = Y_all[:n,:]
    X_test = X_all[n:,:]
    Y_test = Y_all[n:,:]

    lab_train = class_labs[:n]
    lab_test = class_labs[n:]

    return X, Y, X_test, Y_test, lab_train, lab_test


def data_gen_syn(n, n_test, d_x, d_z, d_y, link="linear"):

    X_all, Y_all, Z_all = gen_(n,n_test,d_x,d_z,d_y,link)

    Z = Z_all[:n,:]
    Z_test = Z_all[n:,:]

        
    X = X_all[:n,:]
    Y = Y_all[:n,:]
    X_test = X_all[n:,:]
    Y_test = Y_all[n:,:]

    return X, Y, X_test, Y_test



def top_k_err(X, Y, k):
    X_ = F.normalize(X)
    Y_ = F.normalize(Y)
    logits_mat = torch.matmul(X_,Y_.T)
    topk_list = [logits_mat[i,:].topk(k).indices for i in range(logits_mat.shape[0])]
    topk_listT = [logits_mat.T[i,:].topk(k).indices for i in range(logits_mat.shape[1])]
    err_1 = np.array([i not in topk_list[i] for i in range(len(topk_list))])
    err_2 = np.array([i not in topk_listT[i] for i in range(len(topk_listT))])

    return np.mean(err_1 * err_2)

def match_acc(X, Y, lab1, lab2):
    X_ = F.normalize(X)
    Y_ = F.normalize(Y)
    logits_mat = torch.matmul(X_,Y_.T)
    match_list1 = [logits_mat[i,:].topk(1).indices[0] for i in range(logits_mat.shape[0])]
    match_list2 = [logits_mat[:,j].topk(1).indices[0] for j in range(logits_mat.shape[1])]
#     acc = np.mean([lab1[i] == lab2[j] for i, j in zip(match_list1, match_list2)])
    acc_1 = np.array([lab1[i] == lab1[match_list1[i]] for i in range(len(match_list1))])
    acc_2 = np.array([lab2[i] == lab2[match_list2[i]] for i in range(len(match_list2))])
    acc = np.mean(acc_1 * acc_2)
    
    return acc_1, acc_2


def cartesian_to_polar_batch(vectors):
    """
    Convert a 2D array of n-dimensional vectors from Cartesian to polar coordinates.
    
    Parameters:
    vectors (numpy array): A 2D numpy array where each row is an n-dimensional vector in Cartesian coordinates.
    
    Returns:
    tuple: A tuple where the first element is an array of radii, and the second element is a 2D array of angles.
    """
    num_vectors = vectors.shape[0]
    n = vectors.shape[1]
    
    
    # Compute the radius for each vector (Euclidean norm of each row)
    radii = np.linalg.norm(vectors, axis=1)
    
    # Initialize an array to store angles for each vector
    angles = np.zeros((num_vectors, n - 1))
    
    # Calculate angles for each vector
    for i in range(num_vectors):
        for j in range(n - 1):
            angle = np.arctan2(np.linalg.norm(vectors[i, j+1:]), vectors[i, j])
            angles[i, j] = angle
            
    return radii, angles

    
"""
Layer-wise adaptive rate scaling for SGD in PyTorch!
Based on https://github.com/noahgolmant/pytorch-lars
"""
import torch
from torch.optim.optimizer import Optimizer

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])