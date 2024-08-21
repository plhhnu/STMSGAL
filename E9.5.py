import warnings

warnings.filterwarnings("ignore")
import datetime

now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os

import csv
import ot
import sklearn.metrics as metrics
import STMSGAL

from s_dbw import S_Dbw

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_root = f'/data/pxh/SEDR/data'

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type
label = 0
alpha = 0
dsc_alpha = 0.05
d = 12
pre_resolution = 0.2
n_epochs = 100
reg_ssc = 1
cost_ssc = 0.1
refinement = False
parameter = ''
label = 0
n_clusters = 20
method = 'louvain'

data_name = 'E9.5_E1S1.MOSTA'

adata = sc.read(f'{data_root}/{data_name}.h5ad')

adata.obs.head()
# Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Constructing the spatial network
# STMSGAL.Cal_Spatial_Net(adata, rad_cutoff=3)
STMSGAL.Cal_Spatial_Net(adata, model='KNN', k_cutoff=5)

adata, pred_dsc = STMSGAL.train_STMSGAL(adata, alpha=alpha, n_epochs=n_epochs,
                                        reg_ssc_coef=reg_ssc,
                                        cost_ssc_coef=cost_ssc, n_cluster=n_clusters,
                                        dsc_alpha=dsc_alpha, d=d, category=data_name)
sc.pp.neighbors(adata, use_rep='STMSGAL')
sc.tl.umap(adata)

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.05, end=2, increment=0.005):

    print('Searching resolution...')
    label = 0
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            # print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            # print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    # assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."
    return res, label


if method == 'louvain':
    eval_resolution, label = search_res(adata, n_clusters=n_clusters, method=method)
    sc.tl.louvain(adata, resolution=eval_resolution)
    y = adata.obs['louvain']
    if refinement:
        new_type = refine_label(adata, 50, key=method)
        adata.obs['domain'] = new_type
        y = adata.obs['domain']
    y = y.values.reshape(-1).astype(int)
    n_cluster = len(np.unique(np.array(y)))
    parameter = f"{method},n_cluster{n_cluster},n_epochs={n_epochs}" \
                f"reg_ssc={reg_ssc},cost_ssc = {cost_ssc}" \
                f",dsc_alpha{dsc_alpha},d{d},test"

X = adata.obsm['STMSGAL']
y = adata.obs[method]
y = y.values.reshape(-1)
y = y.codes
n_cluster = len(np.unique(np.array(y)))

dav = np.round(metrics.davies_bouldin_score(X, y), 5)
cal = np.round(metrics.calinski_harabasz_score(X, y), 5)
sil = np.round(metrics.silhouette_score(X, y), 5)
sdbw = np.round(S_Dbw(X, y), 5)

now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if not os.path.exists(f'./outputs/Mouse_Embryo'):
    os.makedirs(f'./outputs/Mouse_Embryo')
with open(f'outputs/Mouse_Embryo/Mouse_Embryo_index.csv', mode='a+') as f:
    f_writer = csv.writer(
        f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f_writer.writerow([str(now)])
    f_writer.writerow([str(data_name)])
    f_writer.writerow([str(parameter)])
    f_writer.writerow(["dav:" + str(dav)])
    f_writer.writerow(["cal:" + str(cal)])
    f_writer.writerow(["sil:" + str(sil)])
    f_writer.writerow(["sdbw:" + str(sdbw)])
    f_writer.writerow([str(" ")])

plt.rcParams["figure.figsize"] = (3, 3)
adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
plt.rcParams["figure.figsize"] = (3, 3)

if data_name == "E9.5_E1S1.MOSTA":
    s=20
else:
    s=30
sc.pl.embedding(adata, basis="spatial",
                color=method,
                s=s,
                show=False,
                # palette=plot_color,
                title='STMSGAL')
plt.savefig(f'./outputs/Mouse_Embryo/{data_name}_{parameter}.jpg',
            bbox_inches='tight', dpi=300)

