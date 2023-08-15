import warnings
warnings.filterwarnings("ignore")
import datetime
now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间1:", now1)
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

category = "Adult Mouse Brain (FFPE)"
adata = sc.read(f"./h5ad/{category}/"
                "final_Adult Mouse Brain (FFPE).h5ad")

adata.uns['log1p']['base'] = None

# # denoise,Itpka,Lrrtm4,Prkcd,Hs3st4,Bcl11b,Syne1,Ddn,Camk2a,Trf,Mobp,Amigo2，Enpp2
# gene = ['S100A11']
# for id in gene:
#     plot_gene = id
#     plt.rcParams["figure.figsize"] = (3, 3)
#     # fig, axs = plt.subplots(1, 2, figsize=(3, 3))
#     sc.pl.spatial(adata, img_key="hires",size=1.2,color=plot_gene, show=False, title=plot_gene, vmax='p99')
#     #sc.pl.spatial(adata, img_key="hires", size=1.2,color=plot_gene, show=False, ax=axs[1], title='MSGATE_'+plot_gene, layer='STMSGAL_ReX', vmax='p99')
#     plt.savefig(f'./outputs/{category}/{plot_gene}.jpg',
#                 bbox_inches='tight', dpi=300)

# #violin
# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# marker_genes = ['Lrrtm4', 'Itpka', 'Hs3st4']
# sc.pl.stacked_violin(adata, marker_genes, ax=axs[0], title='RAW_',roupby='louvain', groupby='louvain',rotation=90,show=False,var_group_labels='5')
# sc.pl.stacked_violin(adata, marker_genes, ax=axs[1],layer='STMSGAL_ReX',groupby='louvain', rotation=90,show=False,var_group_labels='5')
# plt.savefig(f'./outputss/{category}/a_violin_{marker_genes}.jpg',
#             bbox_inches='tight', dpi=300)

# #wilcoxon or t-test
# sc.tl.rank_genes_groups(adata, 'louvain', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False,show=False)
# plt.savefig(f'./outputss/{category}/makergene_t-test.jpg',
#             bbox_inches='tight', dpi=300)

# #heatmap
# sc.tl.rank_genes_groups(adata, 'louvain', method="t-test")
# #14','15','19',,'2','4','13'
# groups=['0']
# sc.pl.rank_genes_groups_heatmap(adata,groups=groups, n_genes=5, groupby='louvain',show=False)
#
# plt.savefig(f'./repaint/{category}/makergene_t-test_groups{groups}.jpg',
#             bbox_inches='tight', dpi=300)

#rank_genes_groups_stacked_violin
sc.tl.rank_genes_groups(adata, 'louvain', method="t-test")
# groups=['12','14','15','19']
groups=['0','4','9']
sc.pl.rank_genes_groups_stacked_violin(adata,groups=groups, n_genes=3,
                                       groupby='louvain',show=False)
plt.savefig(f'./repaint/{category}/a_stacked_violin_t-test_groups{groups}.jpg',
            bbox_inches='tight', dpi=300)
# sc.tl.rank_genes_groups(adata, 'leiden', groups=['4'],
#                         reference='13', method='t-test')
# plt.savefig(f'./outputss/{category}/a_stacked_violin_t-test_reference{groups}.jpg',
#             bbox_inches='tight', dpi=300)
