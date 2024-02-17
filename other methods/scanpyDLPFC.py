
import warnings
warnings.filterwarnings("ignore")
import datetime
now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间1:", now1)
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import random
import sys
import sklearn.metrics as metrics
from s_dbw import S_Dbw
from collections import Counter
from natsort import natsorted
import STAGATE
import os
import eval
import statistics

os.environ["CUDA_VISIBLE_DEVICES"]= "1"
proj_list = ['151507', '151508', '151509', '151510',
             '151669', '151670', '151671', '151672',
             '151673', '151674', '151675', '151676']

ARI_list = []
NMI_list = []

for proj_idx in range(len(proj_list)):
    ARI_list = []
    NMI_list = []
    for b in range(1):
        # category = "V1_Adult_Mouse_Brain_Coronal_Section_1"
        data_name = proj_list[proj_idx]

        if data_name in ['151669', '151670', '151671', '151672']:
            n_domains = 5
        else:
            n_domains = 7

        data_root = f"/data/pxh/SEDR/data/DLPFC/{data_name}"
        adata = sc.read_visium(f"{data_root}")
        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        if not os.path.exists(f'./outputs/{data_name}'):
            os.makedirs(f'./outputs/{data_name}')
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata)

        def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
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

        eval_resolution = res_search_fixed_clus(adata, n_domains)
        sc.tl.louvain(adata, resolution=eval_resolution)
        sc.tl.umap(adata)

        # add ground_truth
        df_meta = pd.read_csv(data_root + f'/metadata.tsv', sep='\t')
        df_meta_layer = df_meta['layer_guess']
        adata.obs['ground_truth'] = df_meta_layer.values

        # filter out NA nodes
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]
        y = adata.obs['louvain']
        n_cluster = len(np.unique(np.array(adata.obs['louvain'])))
        ARI = np.round(metrics.adjusted_rand_score(y, adata.obs['ground_truth']), 5)
        NMI = np.round(metrics.normalized_mutual_info_score(y, adata.obs['ground_truth']), 5)

        ARI_list.append(ARI)
        NMI_list.append(NMI)
        parameter = f"scanpy"
        if not os.path.exists(f'./outputs/DLPFC/{data_name}'):
            os.makedirs(f'./outputs/DLPFC/{data_name}')
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.pl.spatial(adata, img_key="hires", color="louvain", size=1.2,show=False,title='SCANPY')
        plt.savefig(f'./outputs/DLPFC/{data_name}/size=1.2,{parameter}.jpg',
                    bbox_inches='tight', dpi=300)
        sc.pl.spatial(adata, img_key="hires", color="louvain", size=1.5,show=False,title='SCANPY')
        plt.savefig(f'./outputs/DLPFC/{data_name}/size=1.5,{parameter}.jpg',
                    bbox_inches='tight', dpi=300)
        
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.pl.umap(adata, color='louvain', legend_loc='on data', s=20,show=False,
                   title='SCANPY',legend_fontoutline=2)
        plt.savefig(f'./outputs/DLPFC/{data_name}/umap,{parameter}.jpg',
                    bbox_inches='tight', dpi=300)



