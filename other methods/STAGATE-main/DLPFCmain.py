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
from sklearn.metrics.cluster import adjusted_rand_score
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import STAGATE

#data_root = '/data/pxh2/stagate/data/DLPFC'
data_root = '/data/pxh/SEDR/data/DLPFC'
# all DLPFC folder list,'151507', '151508',
proj_list = ['151507', '151508', '151509', '151510',
             '151669', '151670', '151671', '151672',
             '151673', '151674', '151675', '151676']

ARI_list = []
NMI_list = []

for proj_idx in range(len(proj_list)):
    data_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx + 1) + ' : ' + data_name)
    if data_name in ['151669', '151670', '151671', '151672']:
        n_domains = 5
    else:
        n_domains = 7

    file_fold = f'{data_root}/{data_name}'
    save_fold = os.path.join('./outputs/DLPFC', data_name)
    #read data
    adata = sc.read_visium(os.path.join(data_root, data_name))
    adata.var_names_make_unique()

    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Constructing the spatial network
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    STAGATE.Stats_Spatial_Net(adata)

    adata = STAGATE.train_STAGATE(adata, alpha=0)

    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)

    adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=n_domains)

    # add ground_truth
    df_meta = pd.read_csv(data_root + f'/{data_name}/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values

    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    X = adata.obsm['STAGATE']
    y = adata.obs['mclust']

    parameter = ""
    # calculate metric ARI
    ARI = np.round(adjusted_rand_score(y, adata.obs['ground_truth']), 5)
    adata.uns['ARI'] = ARI
    ARI_list.append(ARI)
    print('Dataset:', data_name)
    print('Adjusted rand index = %.3f' % ARI)

    if not os.path.exists(f'./outputs/DLPFC/{data_name}'):
        os.makedirs(f'./outputs/DLPFC/{data_name}')
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(adata, img_key="hires", color="mclust", legend_loc='right margin',
                  size=1.5, show=False)
    plt.savefig(f'./outputs/DLPFC/{data_name}/domain{ARI}.jpg',
                bbox_inches='tight', dpi=300)

    plt.rcParams["figure.figsize"] = (3, 3)

    sc.pl.umap(adata, color="mclust", legend_loc="on data", legend_fontsize='large',
               legend_fontoutline=2, add_outline=False, s=20, show=False, title="")
    plt.savefig(f'./outputs/DLPFC/{data_name}/umap{ARI}.jpg',
                bbox_inches='tight', dpi=300)

    print(adata.isbacked)
    if not os.path.exists(f'./h5ad/DLPFC'):
        os.makedirs(f'./h5ad/DLPFC')
    adata.filename = f'./h5ad/DLPFC/{data_name}-ARI{ARI}.h5ad'
    print(adata.isbacked)
    now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("时间2:", now2)

import csv
with open(f'./outputs/DLPFC/ARIlist.csv', mode='a+') as f:
    f_writer = csv.writer(
        f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f_writer.writerow([str("  ")])
    f_writer.writerow([str(now)])
    f_writer.writerow(["ARI_list", str(ARI_list)])
    f_writer.writerow(["ARI_medium:" + str((sorted(ARI_list)[5] + sorted(ARI_list)[6]) / 2)])
    # f_writer.writerow([str(data_name),"ARI:" + str(ARI)])
    f_writer.writerow(["parameter", str(parameter)])
print("finish")


