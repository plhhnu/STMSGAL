 
import os
import datetime
now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
import numpy as np
import eval
import csv

# all DLPFC folder list
proj_list = ['151507', '151508', '151509', '151510',
             '151669', '151670', '151671', '151672',
             '151673', '151674', '151675', '151676']

ARI_list = []
NMI_list = []
parameter = ""
for proj_idx in range(len(proj_list)):
	data_path = "/data/pxh/SEDR/data" #### to your path
	#data_name = '151673' #### project name
	data_name = proj_list[proj_idx]
	print('===== Project ' + str(proj_idx + 1) + ' : ' + data_name)
	save_path = "../outputs" #### save path
	if data_name in ['151669', '151670', '151671', '151672']:
		n_domains = 5
	else:
		n_domains = 7


	###### the number of spatial domains.
	pca_n_comps = 200
	pre_epochs = 800
	epochs = 1000


	parameter = f"1pca_n_comps={pca_n_comps},pre_epochs={pre_epochs}," \
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
		# linear_encoder_hidden=[64, 16],
		# linear_decoder_hidden=[64],
		# conv_hidden=[64, 16]
		)

	if not os.path.exists(f'./h5ad/DLPFC/morphology{data_name}.h5ad'):
		adata = deepen._get_adata(data_path, data_name)
		print(adata.isbacked)
		adata.filename = f'./h5ad/DLPFC/morphology{data_name}.h5ad'
		print(adata.isbacked)

	adata = sc.read(f"./h5ad/DLPFC/morphology{data_name}.h5ad")
	adata = deepen._get_augment(adata, adjacent_weight = 0.3, neighbour_k = 4)
	graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="BallTree", k=12)
	adata = deepen._fit(adata, graph_dict, pretrain = True,dim_reduction = True)
	adata = deepen._get_cluster_data(adata, n_domains = n_domains, priori=True) ###### without using prior knowledge, setting priori = False.
	# ######## spatial domains
	# deepen.plot_domains(adata, data_name)
	# ######## UMAP
	# deepen.plot_umap(adata, data_name)

	# read the annotation
	# add ground_truth
	df_meta = pd.read_csv(f'{data_path}/{data_name}/metadata.tsv', sep='\t')
	df_meta_layer = df_meta['layer_guess']
	adata.obs['Ground Truth'] = df_meta_layer.values

	# filter out NA nodes
	adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
	#Ann_df = Ann_df[~pd.isnull(Ann_df[''])]
	y = adata.obs['DeepST_refine_domain']
	#y = y.codes
	ARI = np.round(adjusted_rand_score(y, adata.obs['Ground Truth']),5)
	NMI = np.round(normalized_mutual_info_score(y, adata.obs['Ground Truth']),5)
	print('Adjusted rand index = %.3f' % ARI)
	ARI_list.append(ARI)
	NMI_list.append(NMI)

	# print(adata.isbacked)
	# if not os.path.exists(f'./h5ad/DLPFC'):
	# 	os.makedirs(f'./h5ad/DLPFC')
	# adata.filename = f'./h5ad/DLPFC/{data_name}-ARI{ARI}-{parameter}.h5ad'
	# print(adata.isbacked)

	if not os.path.exists(f'./outputs/DLPFC/{data_name}'):
		os.makedirs(f'./outputs/DLPFC/{data_name}')
	plt.rcParams["figure.figsize"] = (3, 3)
	sc.pl.spatial(adata, img_key="hires", color="DeepST_refine_domain", legend_loc='right margin',
				  size=1.5, show=False)
	plt.savefig(f'./outputs/DLPFC/{data_name}/domain{ARI}-{parameter}.jpg',
				bbox_inches='tight', dpi=300)

	plt.rcParams["figure.figsize"] = (3, 3)
	sc.pp.neighbors(adata, n_neighbors = 15,use_rep='DeepST_embed')
	sc.tl.umap(adata)
	sc.pl.umap(adata, color="DeepST_refine_domain", legend_loc="on data",
			   legend_fontoutline=2,add_outline=False , s=20,show=False,title = "DeepST")
	plt.savefig(f'./outputs/DLPFC/{data_name}/umap{ARI}-{parameter}.jpg',
					bbox_inches='tight', dpi=300)

	print(adata.isbacked)
	if not os.path.exists(f'./h5ad/DLPFC'):
		os.makedirs(f'./h5ad/DLPFC')
	adata.filename = f'./h5ad/DLPFC/{data_name}-ARI{ARI}-{parameter}.h5ad'
	print(adata.isbacked)
	now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')



	with open(f'./outputs/DLPFC/ARI.csv', mode='a+') as f:
		f_writer = csv.writer(
			f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

		f_writer.writerow([str(now)])
		f_writer.writerow([str(data_name), "ARI:" + str(ARI), "NMI:" + str(NMI)])


with open(f'./outputs/DLPFC/ARI.csv', mode='a+') as f:
	f_writer = csv.writer(
		f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	f_writer.writerow([str("  ")])
	f_writer.writerow([str(now)])
	f_writer.writerow(["parameter", str(parameter)])
	f_writer.writerow(["ARI_list", str(ARI_list)])
	f_writer.writerow(["NMI_list", str(NMI_list)])
	f_writer.writerow(["ARI_medium:" + str((sorted(ARI_list)[5] + sorted(ARI_list)[6]) / 2)])
	f_writer.writerow(["NMI_medium:" + str((sorted(NMI_list)[5] + sorted(NMI_list)[6]) / 2)])

print("1")
