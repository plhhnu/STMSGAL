# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 20:15
# @Author  : 彭新怀
# @VERSON:1.0
# @File    : mainBrain.py
# @Description :

import os
import datetime

now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间1:", now1)
from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from s_dbw import S_Dbw
import sklearn.metrics as metrics
import anndata

data_path = "/data/pxh/SEDR/data/"  #### to your path
data_name = 'Ductal_Carcinoma_in_situ'  #### project name
# V1_Adult_Mouse_Brain_Coronal_Section_1,Adult_Mouse_Brain (Coronal),Adult Mouse Brain (FFPE)
# Ductal_Carcinoma_in_situ,V1_Breast_Cancer_Block_A_Section_1
save_path = "../outputs"  #### save path

###### the number of spatial domains.
pca_n_comps = 200
pre_epochs = 800
epochs = 1000

for id in [1]:
    n_domains = 20
    if data_name == 'Ductal_Carcinoma_in_situ':
        n_domains = 12
    parameter = f"{n_domains},{id}pca_n_comps={pca_n_comps}," \
                f"pre_epochs={pre_epochs},epochs={epochs}"
    deepen = run(save_path=save_path,
                 platform="Visium",
                 # pca_n_comps = pca_n_comps,
                 # pre_epochs = pre_epochs, #### According to your own hardware, choose the number of training
                 # epochs = epochs, #### According to your own hardware, choose the number of training
                 # Conv_type="GCNConv", #### you can choose GNN types.
                 pca_n_comps=pca_n_comps,
                 pre_epochs=pre_epochs,  #### According to your own hardware, choose the number of training
                 epochs=epochs,  #### According to your own hardware, choose the number of training
                 Conv_type="GCNConv",  #### you can choose GNN types.

                 )
    if not os.path.exists(f'./h5ad/{data_name}/morphology{data_name}.h5ad'):
        adata = deepen._get_adata(data_path, data_name)
        print(adata.isbacked)
        if not os.path.exists(f'./h5ad/{data_name}'):
            os.makedirs(f'./h5ad/{data_name}')
        adata.filename = f'./h5ad/{data_name}/morphology{data_name}.h5ad'
        print(adata.isbacked)

    adata = sc.read(f"./h5ad/{data_name}/morphology{data_name}.h5ad")
    # adata = deepen._get_adata(data_path, data_name)
    adata = deepen._get_augment(adata, adjacent_weight=0.3, neighbour_k=4, )
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="BallTree", k=12)
    adata = deepen._fit(adata, graph_dict, pretrain=True)
    adata = deepen._get_cluster_data(adata, n_domains=n_domains,
                                     priori=True)  ###### without using prior knowledge, setting priori = False.
    # ######## spatial domains
    deepen.plot_domains(adata, data_name)
    # ######## UMAP
    # deepen.plot_umap(adata, data_name)

    obs_df = adata.obs.dropna()
    X = adata.obsm['DeepST_embed']
    y = obs_df['DeepST_refine_domain']
    y = y.values.reshape(-1)
    y = y.astype(np.int64)
    # y = y.codes

    # metrics
    dav = np.round(metrics.davies_bouldin_score(X, y), 5)
    cal = np.round(metrics.calinski_harabasz_score(X, y), 5)
    sil = np.round(metrics.silhouette_score(X, y), 5)
    sdbw = np.round(S_Dbw(X, y), 5)

    # print(adata.isbacked)
    # if not os.path.exists(f'./h5ad/{data_name}'):
    #     os.makedirs(f'./h5ad/{data_name}')
    # adata.filename = f'./h5ad/{data_name}.h5ad'
    # print(adata.isbacked)

    if not os.path.exists(f'./outputs/{data_name}'):
        os.makedirs(f'./outputs/{data_name}')
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(adata, img_key="hires", color="DeepST_refine_domain", legend_loc='right margin',
                  size=1.5, show=False)
    plt.savefig(f'./outputs/{data_name}/domain-{parameter}.jpg',
                bbox_inches='tight', dpi=300)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='DeepST_embed')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color="DeepST_refine_domain", legend_loc="on data",
               legend_fontoutline=2, add_outline=False, s=20, show=False, title="DeepST")
    plt.savefig(f'./outputs/{data_name}/umap-{parameter}.jpg',
                bbox_inches='tight', dpi=300)
    import eval

    eval.Spatialeval(os.path.join(f"./outputs/{data_name}/", f"{data_name}_index.csv"),
                     X, y, X.shape[1], dav, cal, sil, sdbw, parameter)


    eval.saveH5ad(adata=adata, data_name=data_name, parameter=parameter)
    now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("时间2:", now2)
print("finish")
