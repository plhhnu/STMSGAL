import warnings
warnings.filterwarnings("ignore")
import datetime
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import csv
import ot
import sklearn.metrics as metrics
import STMSGAL
import statistics

data_root = f'/data/pxh/SEDR/data/STRAmap'
label = 0
rad_cutoff = 400
alpha = 0
dsc_alpha = 0.05
d = 12
n_epochs = 400
reg_ssc = 1
cost_ssc = 0.1
parameter = ''
label = 0
n_clusters = 7
method = 'mclust'
id = 0

data_name = 'STARmap_20180505_BY3_1k'

adata = sc.read(f'{data_root}/{data_name}.h5ad')

adata.obs.head()
# Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Constructing the spatial network
STMSGAL.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)

adata, pred_dsc = STMSGAL.train_STAGATE(adata, alpha=alpha, n_epochs=n_epochs,
                                        reg_ssc_coef=reg_ssc,
                                        cost_ssc_coef=cost_ssc, n_cluster=n_clusters,
                                        dsc_alpha=dsc_alpha, d=d, category=data_name)
sc.pp.neighbors(adata, use_rep='STMSGAL')
sc.tl.umap(adata)
ARI = 0.0
NMI = 0.0

adata = STMSGAL.mclust_R(adata, used_obsm='STMSGAL', num_cluster=n_clusters)
ARI = np.round(metrics.adjusted_rand_score(adata.obs[method], adata.obs['label']), 5)
parameter = f"{id}{method},n_cluster{len(np.unique((adata.obs[method])))},n_epochs={n_epochs}" \
            f"reg_ssc={reg_ssc},cost_ssc = {cost_ssc}" \
            f",dsc_alpha{dsc_alpha},d{d}"

if not os.path.exists(f'./outputs/{data_name}'):
    os.makedirs(f'./outputs/{data_name}')
plt.rcParams["figure.figsize"] = (4, 2)
sc.pl.embedding(adata, basis="spatial", color=method, s=20, show=False)
plt.savefig(f'./outputs/{data_name}/ARI{ARI}_{parameter}.jpg',
            bbox_inches='tight', dpi=300)

plt.rcParams["figure.figsize"] = (4, 2)
sc.pl.embedding(adata, basis="spatial", color='label', s=20, show=False)
plt.savefig(f'./outputs/{data_name}/label.jpg',
            bbox_inches='tight', dpi=300)

with open(f'./outputs/{data_name}/ARI.csv', mode='a+') as f:
    f_writer = csv.writer(
        f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f_writer.writerow([str(now)])
    f_writer.writerow([str(parameter)])
    f_writer.writerow([str(data_name), "ARI:" + str(ARI)])
    f_writer.writerow([str(" ")])
print(adata.isbacked)
if not os.path.exists(f'./h5ad2/{data_name}'):
    os.makedirs(f'./h5ad2/{data_name}')
adata.filename = f'./h5ad2/{data_name}/{data_name}_{parameter}.h5ad'
print(adata.isbacked)

