

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

data_root = '/data/pxh/SEDR/data'
# all DLPFC folder list

proj_list = ['V1_Breast_Cancer_Block_A_Section_1']
save_root = './outputs/V1_Breast_Cancer_Block_A_Section_1/'

for proj_idx in range(len(proj_list)):
    data_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx + 1) + ' : ' + data_name)
    file_fold = f'{data_root}/{data_name}'
    # read data
    adata = sc.read_visium(os.path.join(data_root, data_name))
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # add ground_truth
    df_meta = pd.read_csv(data_root + f'/{data_name}/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['fine_annot_type']
    adata.obs['ground_truth'] = df_meta_layer.values

    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    if not os.path.exists(f'./outputs/{data_name}'):
        os.makedirs(f'./outputs/{data_name}')
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(adata, img_key="hires", color='ground_truth', size=1.5,show=False)
    plt.savefig(f'./outputs/{data_name}/{data_name},size=1.5,domain_ground.jpg',
                bbox_inches='tight', dpi=300)
    sc.pl.spatial(adata, img_key="hires", color='ground_truth', size=1.2,show=False)
    plt.savefig(f'./outputs/{data_name}/{data_name},size=1.2,domain_ground.jpg',
                bbox_inches='tight', dpi=300)


data_root = '/data/pxh/SEDR/data/DLPFC'
# all DLPFC folder list
proj_list = ['151507', '151508', '151509', '151510',
             '151669', '151670', '151671', '151672',
             '151673', '151674', '151675', '151676']
# proj_list = ['151676']
save_root = './outputs/DLPFC/'

for proj_idx in range(len(proj_list)):
    data_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx + 1) + ' : ' + data_name)
    file_fold = f'{data_root}/{data_name}'
    # read data
    adata = sc.read_visium(os.path.join(data_root, data_name))
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # add ground_truth
    df_meta = pd.read_csv(data_root + f'/{data_name}/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values

    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    if not os.path.exists(f'./outputs/DLPFC/{data_name}'):
        os.makedirs(f'./outputs/DLPFC/{data_name}')
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(adata, img_key="hires", color='ground_truth', size=1.5,show=False)
    plt.savefig(f'./outputs/DLPFC/{data_name},size=1.5,domain_ground.jpg',
                bbox_inches='tight', dpi=300)
    sc.pl.spatial(adata, img_key="hires", color='ground_truth', size=1.2,show=False)
    plt.savefig(f'./outputs/DLPFC/{data_name},size=1.2,domain_ground.jpg',
                bbox_inches='tight', dpi=300)
