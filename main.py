import warnings
warnings.filterwarnings("ignore")
import datetime
now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("time1:", now1)
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
import STMSGAL
import os
import eval
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

random.seed(1234)
np.random.seed(1234)
#tf.set_random_seed(1234)
#Adult Mouse Brain (FFPE),V1_Adult_Mouse_Brain_Coronal_Section_1,Adult Mouse Kidney (FFPE)
#Adult_Mouse_Brain (Coronal),V1_Breast_Cancer_Block_A_Section_1,Visium_Mouse_Olfactory_Bulb1.3
#Ductal_Carcinoma_in_situ,Human Lymph Node

label = 0

category = "Ductal_Carcinoma_in_situ"
director = f"/data/pxh/SEDR/data/{category}"
adata = sc.read_visium(f"{director}")
adata.var_names_make_unique()

#k_cutoff = 15
rad_cutoff = 300
alpha = 0.5
dsc_alpha = 0.35
d = 6
pre_resolution = 0.2
n_epochs=300
n_clusters = 12
cost_ssc_coef = 1

n_top_genes = 3000
method = 'louvain'

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires",show=False)
if not os.path.exists(f'./outputs/{category}'):
    os.makedirs(f'./outputs/{category}')
plt.savefig(f'./outputs/{category}/hires.jpg', bbox_inches='tight', dpi=150)

#Constructing the spatial network
STMSGAL.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
STMSGAL.Stats_Spatial_Net(adata)
Mean_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]/adata.shape[0]
print(Mean_edge)
#Running
adata ,pred_dsc= STMSGAL.train_STMSGAL(adata, alpha=alpha, pre_resolution=pre_resolution,
                              n_epochs=n_epochs, save_attention=True,save_loss=False,
                              cost_ssc_coef = cost_ssc_coef,dsc_alpha=dsc_alpha, d=d)
# pre-clustering result
if not os.path.exists(f'./outputs/{category}'):
    os.makedirs(f'./outputs/{category}')
if alpha>0:
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(adata, img_key="hires",show=False,
                  color="expression_louvain_label", size=1.2, title='pre-clustering result')
    plt.savefig(f'./outputs/{category}/precluster.jpg', bbox_inches='tight', dpi=300)


sc.pp.neighbors(adata, use_rep='STMSGAL')
sc.tl.umap(adata)
def res_search_fixed_clus(adata, fixed_clus_count ,method='louvain', use_rep='STMSGAL', start=0.1, end=2.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    #sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            #print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            #print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == fixed_clus_count:
            label = 1
            break

    #assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res,label


if method == 'leiden':
    eval_resolution,label = res_search_fixed_clus(adata, n_clusters,method = method
                                            , start=0.3, end=2.5, increment=0.01)
    sc.tl.leiden(adata, key_added="leiden", resolution=eval_resolution)
elif method == 'louvain':
    eval_resolution,label = res_search_fixed_clus(adata, n_clusters,method = method
                                            , start=0.3, end=2.5, increment=0.01)
    sc.tl.louvain(adata, resolution=eval_resolution)

X = adata.obsm['STMSGAL']
y = adata.obs['louvain']
y = y.values.reshape(-1)
y = y.codes
n_cluster = len(np.unique(np.array(y)))

parameter = f" "

dav = np.round(metrics.davies_bouldin_score(X, y), 5)
cal = np.round(metrics.calinski_harabasz_score(X, y), 5)
sil = np.round(metrics.silhouette_score(X, y), 5)
table = []
sdbw = np.round(S_Dbw(X, y), 5)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color=method, size=1.5,show=False,title=f'(α={alpha})')
plt.savefig(f'./outputs/{category}/atest.jpg',
            bbox_inches='tight', dpi=300)
plt.savefig(f'./outputs/{category}/size=1.5_cluster{n_cluster}_{parameter}.jpg',
            bbox_inches='tight', dpi=300)
sc.pl.spatial(adata, img_key="hires", color=method, size=1.2,show=False,title=f'(α={alpha})')
plt.savefig(f'./outputs/{category}/size=1.2_cluster{n_cluster}_{parameter}.jpg',
            bbox_inches='tight', dpi=300)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata, color=method, legend_loc='on data', legend_fontoutline='2',s=20,show=False,title=f'(α={alpha})')
plt.savefig(f'./outputs/{category}/umap{n_cluster}_{parameter}.jpg',
            bbox_inches='tight', dpi=300)

eval.Spatialeval(os.path.join(f"./outputs/{category}/", f"{category}_index.csv"),
                     X, y, X.shape[1], dav, cal, sil, sdbw, table,parameter)

print(adata.isbacked)
if not os.path.exists(f'./h5ad/{category}'):
    os.makedirs(f'./h5ad/{category}')
adata.filename = f'./h5ad/{category}/final_{category}_{parameter}.h5ad'
print(adata.isbacked)
now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("time2:", now2)
print("dav:", dav)
print("cal:", cal)
print("sil:", sil)
print("sdbw:", sdbw)
print("Finish")
