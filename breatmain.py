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
import STMSGAL
import os
import eval
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

random.seed(1234)
np.random.seed(1234)
#tf.set_random_seed(1234)
#Adult Mouse Brain (FFPE),V1_Adult_Mouse_Brain_Coronal_Section_1,Adult Mouse Kidney (FFPE)
#Adult_Mouse_Brain (Coronal)
category = "V1_Breast_Cancer_Block_A_Section_1"
director = f"/data/pxh/SEDR/data/{category}"
adata = sc.read_visium(f"{director}")
adata.var_names_make_unique()

#k_cutoff = 15
rad_cutoff = 300
alpha = 0.7
pre_resolution = 0.2
n_epochs=300
dsc_alpha = 0.35
d = 6

cost_ssc_coef = 0.1
n_top_genes=1000
pca = False

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

#Ground fig
Ann_df = pd.read_csv(f'{director}/metadata.tsv', sep='\t')
# adata.obs['Ground Truth'] = Ann_df['fine_annot_type']
# Y = adata.obs['Ground Truth']
result = pd.DataFrame(Ann_df['fine_annot_type'])
adata.obs['Ground Truth'] = result['fine_annot_type'].values

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"],size=1.5,show=False)
plt.savefig(f'./outputs/{category}/Ground1.5.jpg',
            bbox_inches='tight', dpi=150)

#Constructing the spatial network
STMSGAL.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
STMSGAL.Stats_Spatial_Net(adata)

#Running STMSGAL with cell type-aware module
adata,pred_dsc = STMSGAL.train_STMSGAL(adata, alpha=alpha, pre_resolution=pre_resolution,
                              n_epochs=n_epochs, save_attention=True,save_loss=False,
                              n_cluster = 20,cost_ssc_coef = cost_ssc_coef
                             )

if not os.path.exists(f'./outputs/{category}'):
    os.makedirs(f'./outputs/{category}')

sc.pp.neighbors(adata, use_rep='STMSGAL')
sc.tl.umap(adata)

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.01):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.6, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res
eval_resolution = res_search_fixed_clus(adata, 20)

sc.tl.leiden(adata, resolution=eval_resolution)

#read the annotation
Ann_df = pd.read_csv(f'{director}/metadata.tsv', sep='\t')

X = adata.obsm['STMSGAL']
y = adata.obs['leiden']
y = y.values.reshape(-1)
y = y.codes
n_cluster = len(np.unique(np.array(y)))

parameter = f" "

ARI = np.round(metrics.adjusted_rand_score(y, Ann_df['fine_annot_type']), 4)
NMI = np.round(metrics.normalized_mutual_info_score(y, Ann_df['fine_annot_type']), 4)
import csv

with open(f'./outputs/{category}/ARI_NMI.csv', mode='a+') as f:
    f_writer = csv.writer(
        f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f_writer.writerow([str("  ")])
    f_writer.writerow([str(now)])
    f_writer.writerow(["ARI_list",str(ARI)])
    f_writer.writerow(["NMI_list",str(NMI)])
    f_writer.writerow(["parameter", str(parameter)])

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color="leiden", size=1.2,show=False,title='(α=0.7)')
plt.savefig(f'./outputs/{category}/atest.jpg',
                    bbox_inches='tight', dpi=300)
sc.pl.spatial(adata, img_key="hires", color="leiden", size=1.2,show=False,title='(α=0.7)')
plt.savefig(f'./outputs/{category}/cluster{n_cluster}_{parameter}.jpg',
            bbox_inches='tight', dpi=300)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata, color=['leiden'], legend_loc='on data', legend_fontoutline='2', s=20,show=False,title='(α=0.7)')
plt.savefig(f'./outputs/{category}/umap{n_cluster}_{parameter}.jpg',
            bbox_inches='tight', dpi=300)

print(adata.isbacked)
if not os.path.exists(f'./h5ad/{category}'):
    os.makedirs(f'./h5ad/{category}')
adata.filename = f'./h5ad/{category}/final_{category}_{parameter}.h5ad'
print(adata.isbacked)
now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间2:", now2)
