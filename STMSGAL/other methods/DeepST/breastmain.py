# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 19:24
# @Author  : 彭新怀
# @VERSON:1.0
# @File    : breastmain.py
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
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
import numpy as np
import eval
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# proj_list = ['151507', '151508', '151509', '151510',
#              '151669', '151670', '151671', '151672',
#              '151673', '151674', '151675', '151676']
proj_list = ['V1_Breast_Cancer_Block_A_Section_1']
ARI_list = []
NMI_list = []
parameter = ""
#for proj_idx in range(len(proj_list)):
data_path = "/data/pxh/SEDR/data" #### to your path
data_name = 'V1_Breast_Cancer_Block_A_Section_1' #### project name
# data_name = proj_list[proj_idx]
# print('===== Project ' + str(proj_idx + 1) + ' : ' + data_name)
save_path = "../outputs" #### save path

n_domains = 20

###### the number of spatial domains.
pca_n_comps = 200
pre_epochs = 800
epochs = 1000

parameter = f"2pca_n_comps={pca_n_comps},pre_epochs={pre_epochs}," \
            f"epochs={epochs}"
deepen = run(save_path = save_path,
    platform = "Visium",
     # pca_n_comps=100,
     # pre_epochs=1000,  #### According to your own hardware, choose the number of training
     # epochs=1000,
    pca_n_comps = pca_n_comps,
    pre_epochs = pre_epochs, #### According to your own hardware, choose the number of training
    epochs = epochs, #### According to your own hardware, choose the number of training
    Conv_type="GCNConv", #### you can choose GNN types.

    )

if not os.path.exists(f'./h5ad/{data_name}/morphology{data_name}.h5ad'):
    adata = deepen._get_adata(data_path, data_name)
    if not os.path.exists(f'./h5ad/{data_name}'):
        os.makedirs(f'./h5ad/{data_name}')
    print(adata.isbacked)
    adata.filename = f'./h5ad/{data_name}/morphology{data_name}.h5ad'
    print(adata.isbacked)

adata = sc.read(f"./h5ad/{data_name}/morphology{data_name}.h5ad")
adata = deepen._get_augment(adata, adjacent_weight = 0.3, neighbour_k = 4,)
graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="BallTree", k=12)
adata = deepen._fit(adata, graph_dict, pretrain = True,dim_reduction = True)
adata = deepen._get_cluster_data(adata, n_domains = n_domains, priori=True) ###### without using prior knowledge, setting priori = False.
# ######## spatial domains
# deepen.plot_domains(adata, data_name)
# ######## UMAP
# deepen.plot_umap(adata, data_name)

#read the annotation
# add ground_truth
df_meta = pd.read_csv(f'{data_path}/{data_name}/metadata.tsv', sep='\t')
df_meta_layer = df_meta['fine_annot_type']
adata.obs['ground_truth'] = df_meta_layer.values
# filter out NA nodes
adata = adata[~pd.isnull(adata.obs['ground_truth'])]

y = adata.obs['DeepST_refine_domain']

ARI = np.round(adjusted_rand_score(y, adata.obs['ground_truth']),5)
NMI = np.round(normalized_mutual_info_score(y, adata.obs['ground_truth']),5)
print('Adjusted rand index = %.3f' % ARI)
ARI_list.append(ARI)
NMI_list.append(NMI)

if not os.path.exists(f'./outputs/{data_name}'):
    os.makedirs(f'./outputs/{data_name}')
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color="DeepST_refine_domain", legend_loc='right margin',
              size=1.5, show=False)
plt.savefig(f'./outputs/{data_name}/domain{ARI}-{parameter}.jpg',
            bbox_inches='tight', dpi=300)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pp.neighbors(adata, n_neighbors = 15,use_rep='DeepST_embed')
sc.tl.umap(adata)
sc.pl.umap(adata, color="DeepST_refine_domain", legend_loc="on data",
           legend_fontoutline=2,add_outline=False , s=20,show=False,title = "DeepST")
plt.savefig(f'./outputs/{data_name}/umap{ARI}-{parameter}.jpg',
                bbox_inches='tight', dpi=300)

print(adata.isbacked)
if not os.path.exists(f'./h5ad/{data_name}'):
    os.makedirs(f'./h5ad/{data_name}')
adata.filename = f'./h5ad/{data_name}/{data_name}-ARI{ARI}-{parameter}.h5ad'
print(adata.isbacked)
now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间2:", now2)
import csv
with open(f'./outputs/{data_name}/ARI.csv', mode='a+') as f:
    f_writer = csv.writer(
        f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    f_writer.writerow([str(now)])
    f_writer.writerow([str(data_name), "ARI:" + str(ARI), "NMI:" + str(NMI)])

print("1")