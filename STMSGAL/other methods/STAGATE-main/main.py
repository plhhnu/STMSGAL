import warnings
warnings.filterwarnings("ignore")
import datetime
now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间1:", now1)
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import sklearn.metrics as metrics
from s_dbw import S_Dbw
from collections import Counter
import eval

import STAGATE

proj_list = ['Adult Mouse Brain (FFPE)']
for proj_idx in range(len(proj_list)):
    category = proj_list[proj_idx]
    rad_cutoff=300
    alpha = 0.5
    pre_resolution = 0.2
    n_epochs=1000
    method = 'louvain'
    num_cluster = 20

    director = f"../SEDR/data/{category}"
    adata = sc.read_visium(f"{director}")
    adata.var_names_make_unique()

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


    def res_search_fixed_clus(adata, fixed_clus_count, increment=0.005):
        '''
            arg1(adata)[AnnData matrix]
            arg2(fixed_clus_count)[int]

            return:
                resolution[int]
        '''
        for res in sorted(list(np.arange(0.3, 2, increment)), reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            if count_unique_louvain == fixed_clus_count:
                break
        return res

    #Constructing the spatial network
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    STAGATE.Stats_Spatial_Net(adata)

    #Running STAGATE with cell type-aware module
    adata = STAGATE.train_STAGATE(adata, alpha=alpha, pre_resolution=pre_resolution,
                                  n_epochs=n_epochs, save_attention=True, save_reconstrction=False)

    # # pre-clustering result
    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.spatial(adata, img_key="hires",
    #               color="expression_louvain_label", size=1.5, title='pre-clustering result',show=False)
    #
    # plt.savefig(f'./outputs/{category}/precluster.jpg', bbox_inches='tight', dpi=150)
    if not os.path.exists(f'./outputs/{category}'):
        os.makedirs(f'./outputs/{category}')
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)

    resolution = res_search_fixed_clus(adata, num_cluster)
    # resolution = 1
    sc.tl.louvain(adata, resolution=resolution)

    parameter = f"rad_cutoff={rad_cutoff},alpha={alpha},{method}" \
                f"pre_resolution={pre_resolution},n_epochs={n_epochs},resolution={resolution}"

    X = adata.obsm['STAGATE']
    y = adata.obs[method]
    y = y.values.reshape(-1)
    y = y.codes
    n_cluster = len(np.unique(np.array(y)))

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(adata, img_key="hires", color=method, size=1.2,show=False,title='STAGATE')
    plt.savefig(f'./outputs/{category}/size1.2_cluster{n_cluster}_{parameter}.jpg',
                bbox_inches='tight', dpi=300)
    sc.pl.spatial(adata, img_key="hires", color=method, size=1.5,show=False,title='STAGATE')
    plt.savefig(f'./outputs/{category}/size1.5_cluster{n_cluster}_{parameter}.jpg',
                bbox_inches='tight', dpi=300)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=method, legend_loc='on data', s=20,legend_fontoutline='2'
               ,show=False,title='STAGATE')
    plt.savefig(f'./outputs/{category}/umap{n_cluster}_{parameter}.jpg',
                bbox_inches='tight', dpi=300)

    dav = np.round(metrics.davies_bouldin_score(X, y), 5)
    cal = np.round(metrics.calinski_harabasz_score(X, y), 5)
    sil = np.round(metrics.silhouette_score(X, y), 5)
    sdbw = np.round(S_Dbw(X, y), 5)
    table = []

    now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("时间2:", now2)
    print("dav:", dav)
    print("cal:", cal)
    print("sil:", sil)
    print("sdbw:", sdbw)

