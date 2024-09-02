import torch
import time
import random
import numpy as np
import pandas as pd
import os
import psutil
import math
import scanpy as sc
import scanpy.external as sce
import anndata
from pathlib import Path
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix, csc_matrix
from scipy.spatial import distance
from scipy import sparse
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm
from typing import Union, Callable

from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
from .model import GraphCVAEEncoder
from .utils_func import *
from .his_feat import image_feature, image_crop
from .adj import graph, combine_graph_dict
from .augment import augment_adata


class run():
	def __init__(
		self,
		save_path="./",
		):
		self.save_path = save_path
	def _get_adata(
		self,
		platform, 
		data_path,
		data_name,
		verbose = True,
		):
		assert platform in ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
		if platform in ['Visium', 'ST']:
			if platform == 'Visium':
				adata = read_10X_Visium(os.path.join(data_path, data_name))
			else:
				adata = ReadOldST(os.path.join(data_path, data_name))
		elif platform == 'MERFISH':
			adata = read_merfish(os.path.join(data_path, data_name))
		elif platform == 'slideSeq':
			adata = read_SlideSeq(os.path.join(data_path, data_name))
		elif platform == 'seqFish':
			adata = read_seqfish(os.path.join(data_path, data_name))
		elif platform == 'stereoSeq':
			adata = read_stereoSeq(os.path.join(data_path, data_name))
		else:
			raise ValueError(
               				 f"""\
               				 {self.platform!r} does not support.
	                				""")
		if verbose:
			save_data_path = Path(os.path.join(self.save_path, "Data", data_name))
			save_data_path.mkdir(parents=True, exist_ok=True)
			adata.write(os.path.join(save_data_path, f'{data_name}_raw.h5ad'), compression="gzip")
		return adata

	def _get_image_crop(
		self,
		adata,
		data_name,
		cnnType = 'ResNet50',
		pca_n_comps = 50, 
		):
		save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', data_name))
		save_path_image_crop.mkdir(parents=True, exist_ok=True)
		adata = image_crop(adata, save_path=save_path_image_crop)
		adata = image_feature(adata, pca_components = pca_n_comps, cnnType = cnnType).extract_image_feat()
		return adata

	def _get_augment(
		self,
		adata,
		adjacent_weight = 0.3,
		neighbour_k = 4,
		spatial_k = 30,
		n_components = 100,
		md_dist_type="cosine",
		gb_dist_type="correlation",
		use_morphological = True,
		use_data = "raw",
		spatial_type = "KDTree"
		):
		adata = augment_adata(adata, 
				md_dist_type = md_dist_type,
				gb_dist_type = gb_dist_type,
				n_components = n_components,
				use_morphological = use_morphological,
				use_data = use_data,
				neighbour_k = neighbour_k,
				adjacent_weight = adjacent_weight,
				spatial_k = spatial_k,
				spatial_type = spatial_type
				)
		print("Augment molecule expression is Done!")
		return adata

def adj_to_edge_index(adj):
    edge_index = (adj != 0).nonzero().t().contiguous()
    return edge_index
class GraphCVAE():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.002,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=520, 
        dim_input=3000,
        dim_output=64,
        random_seed = 273,
        alpha = 10,
        beta = 1,
        theta = 0.2,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X',
        n_top_genes=2000
        ):
        '''\

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        adata_sc : anndata, optional
            AnnData object of scRNA-seq data. adata_sc is needed for deconvolution. The default is None.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning. 
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning. 
            The default is 1.

        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        
        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata,n_top_genes=2000)
        
        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata)
           else:    
              construct_interaction(self.adata)
         
        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
    
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
        
        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs
            
    def train(self):
        edge_index = adj_to_edge_index(self.graph_neigh)

        if self.datatype in ['Stereo', 'Slide']:
           self.model = GraphCVAEEncoder(self.dim_input, self.dim_output, edge_index).to(self.device)
        else:
            
            self.model = GraphCVAEEncoder(self.dim_input, self.dim_output, edge_index).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
        # self.loss_CSL = nn.CrossEntropyLoss()

    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        print('Begin to train ST data...')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()

            # Generate a perturbed version of features for contrastive learning
            features_a = permutation(self.features)
            # Run the model and obtain necessary outputs
            mu, logvar, hiden_feat, emb, ret, ret_a = self.model(self.features, features_a)

            # Calculate the reconstruction loss and KL divergence loss
            loss_recon = F.mse_loss(self.features, emb)
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Calculate the contrastive learning losses
            loss_contrastive_1 = self.loss_CSL(ret, self.label_CSL)
            loss_contrastive_2 = self.loss_CSL(ret_a, self.label_CSL)

            # Total loss includes reconstruction loss, KL divergence loss, and contrastive learning losses
            total_loss = self.alpha * loss_recon + self.beta * (loss_contrastive_1 + loss_contrastive_2) + self.theta * loss_kl

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    
        
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
             self.model.eval()
             if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a)[1]
                
                return self.emb_rec
             else:  
                if self.datatype in ['Stereo', 'Slide']:
                   self.emb_rec = self.model(self.features, self.features_a)[1]
                   self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy() 
                else:
                   self.emb_rec = self.model(self.features, self.features_a)[1].detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec
                
                return self.adata
         
    
    def loss(self, emb_sp, emb_sc):
        '''\
        Calculate loss

        Parameters
        ----------
        emb_sp : torch tensor
            Spatial spot representation matrix.
        emb_sc : torch tensor
            scRNA cell representation matrix.

        Returns
        -------
        Loss values.

        '''
        # cell-to-spot
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.

        Returns
        -------
        loss : float
            Loss value.

        '''
        
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M        
