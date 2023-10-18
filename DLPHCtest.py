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
import csv
import ot
import sklearn.metrics as metrics
import STMSGAL
import statistics
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

data_root = '/data/pxh/SEDR/data/DLPFC'
# data_root = '/data/pxh2/STMSGAL/data/DLPFC'
# all DLPFC folder list
proj_list = ['151507', '151508', '151509', '151510',
             '151669', '151670', '151671', '151672',
             '151673', '151674', '151675', '151676']
#proj_list = ['151509', '151509', '151510']
# set saving result path
save_root = './outputs/DLPFC/'
ARI_list = []
NMI_list = []
ARI_average_list = []
NMI_average_list = []
ARI_medium_list = []
NMI_medium_list = []

rad_cutoff = 150
alpha = 0
pre_resolution = 0.2
n_epochs = 200

cost_ssc = 0.1

method = 'louvain'
refinement = True
parameter = ''
data_name = ''

n_top_genes = 3000
a = [0]
label = 0

id = 25
for proj_idx in range(len(proj_list)):
    ARI_list = []
    NMI_list = []

    data_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx + 1) + ' : ' + data_name)
    if data_name in ['151669', '151670', '151671', '151672']:
        n_clusters = 5
    else:
        n_clusters = 7

    file_fold = f'{data_root}/{data_name}'
    #read data
    adata = sc.read_visium(os.path.join(data_root, data_name))
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")

    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Constructing the spatial network
    STMSGAL.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)

    adata ,pred_dsc= STMSGAL.train_STMSGAL(adata, alpha=alpha,n_epochs=n_epochs,
                                               reg_ssc_coef=reg_ssc,
                                               cost_ssc_coef=cost_ssc,n_cluster = n_clusters)

    sc.pp.neighbors(adata, use_rep='STMSGAL')
    sc.tl.umap(adata)

    def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.05, end=2.5, increment=0.005):

        print('Searching resolution...')
        label = 0
        for res in sorted(list(np.arange(start, end, increment)), reverse=True):
            if method == 'leiden':
                sc.tl.leiden(adata, random_state=0, resolution=res)
                count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
                #print('resolution={}, cluster number={}'.format(res, count_unique))
            elif method == 'louvain':
                sc.tl.louvain(adata, random_state=0, resolution=res)
                count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
                #print('resolution={}, cluster number={}'.format(res, count_unique))
            if count_unique == n_clusters:
                label = 1
                break
        #assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."
        return res,label

    if method == 'leiden':
        eval_resolution,label = search_res(adata, n_clusters = n_clusters, method = method)
        sc.tl.leiden(adata, resolution=eval_resolution)
    elif method == 'louvain':
        eval_resolution,label = search_res(adata, n_clusters=n_clusters, method=method)
        sc.tl.louvain(adata, resolution=eval_resolution)
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
        cluster_labels = cluster_model.fit_predict(adata.obsm['STMSGAL'])

        adata.obs['kmeans'] = cluster_labels

    # add ground_truth
    df_meta = pd.read_csv(data_root + f'/{data_name}/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values

    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    if refinement:
        new_type = STMSGAL.utils.refine_label(adata, 50, key=method)
        adata.obs['domain'] = new_type

    y = adata.obs['domain']
    y = y.values.reshape(-1).astype(int)

    n_cluster = len(np.unique(np.array(y)))
    ARI = np.round(metrics.adjusted_rand_score(y, adata.obs['ground_truth'] ), 5)
    NMI = np.round(metrics.normalized_mutual_info_score(y, adata.obs['ground_truth']), 5)

    if not os.path.exists(f'./outputs/DLPFC/{data_name}'):
        os.makedirs(f'./outputs/DLPFC/{data_name}')
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(adata, img_key="hires", color='domain', size=1.5,show=False)
    plt.savefig(f'./outputs/DLPFC/{data_name}/ARI{ARI}_cluster{n_cluster}_{parameter}.jpg',
                bbox_inches='tight', dpi=300)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color='domain', legend_loc='on data', s=20,legend_fontoutline='2',
               show=False)
    plt.savefig(f'./outputs/DLPFC/{data_name}/ARI{ARI}_umap{n_cluster}_{parameter}.jpg',
                bbox_inches='tight', dpi=300)
    ARI_list.append(ARI)
    NMI_list.append(NMI)
    # print(f"dataset:{data_name}")
    # print(f"ARI={ARI},NMI={NMI}")

    with open(f'{save_root}/test.csv', mode='a+') as f:
        f_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        f_writer.writerow([str(now)])
        f_writer.writerow([str(parameter)])
        f_writer.writerow([str(data_name),"ARI:" + str(ARI), "NMI:" + str(NMI)])
        f_writer.writerow([str(" ")])
    print(adata.isbacked)
    if not os.path.exists(f'./h5ad2/DLPFC/{data_name}'):
        os.makedirs(f'./h5ad2/DLPFC/{data_name}')
    adata.filename = f'./h5ad2/DLPFC/{data_name}/{data_name}_{parameter}.h5ad'
    print(adata.isbacked)

now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间2:", now2)
print("finish")

