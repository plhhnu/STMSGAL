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
import eval

dataset = ["Adult Mouse Brain (FFPE)", "Adult Mouse Kidney (FFPE)",
            "Ductal_Carcinoma_in_situ",
           "V1_Adult_Mouse_Brain_Coronal_Section_1", "Human Lymph Node"]
dataset = ["V1_Breast_Cancer_Block_A_Section_1","Ductal_Carcinoma_in_situ",
           "V1_Adult_Mouse_Brain_Coronal_Section_1", "Human Lymph Node"]
dataset = ["Ductal_Carcinoma_in_situ"]

alpha0 = [0,0.3,0.7,1]
#重画不同alpha的图
for i in dataset:
    category = i
    alpha = 0.5
    if not os.path.exists(f'./repaint/{category}'):
        os.makedirs(f'./repaint/{category}')
    adata = sc.read(f"./repaint/{category}/"
                    f"final_{i}.h5ad")

    X = adata.obsm['STMSGAL']
    # X = adata.obsm['X_pca']
    y = adata.obs['louvain']
    # y = adata.obs['leiden']

    y = y.values.reshape(-1)
    y = y.codes
    n_cluster = len(np.unique(np.array(y)))

    plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.spatial(adata, img_key="hires",groups=['18'],color="louvain", size=1.5,show=False,title='MSGATE')

    sc.pl.spatial(adata, img_key="hires", color="louvain", size=1.5, show=False,
                  title=f'MSGATE(α={alpha})')
    plt.savefig(f'./repaint/{category}/atest.jpg',
                bbox_inches='tight', dpi=300)

    plt.savefig(f'./repaint/{category}/size=1.5_α={alpha}_{category}.jpg',
                bbox_inches='tight', dpi=300)
    # sc.pl.spatial(adata, img_key="hires", color="louvain", size=1.2,show=False,title='MSGATE(α=1)')
    sc.pl.spatial(adata, img_key="hires", color="louvain", size=1.2, show=False,
                  title=f'MSGATE(α={alpha})')
    plt.savefig(f'./repaint/{category}/size=1.2_α={alpha}_{category}.jpg',
                bbox_inches='tight', dpi=300)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color="louvain", legend_loc='on data', legend_fontoutline='2',
               s=20, title=f'MSGATE(α={alpha})', show=False)
    plt.savefig(f'./repaint/{category}/umap_α={alpha}_{category}.jpg',
                bbox_inches='tight', dpi=300)

    now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("时间2:", now2)


