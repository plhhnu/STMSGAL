#import matplotlib as mpl
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import datetime
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

category = "Adult Mouse Brain (FFPE)"
adata = sc.read(f"./h5ad2/{category}/"
                "final_Adult Mouse Brain (FFPE).h5ad")

att_df = pd.DataFrame(adata.uns['STMSGAL_attention'], index=adata.obs_names, columns=adata.obs_names)
att_df = att_df.values
for it in range(att_df.shape[0]):
    att_df[it, it] = 0

G_atten = nx.from_numpy_matrix(att_df)
M = G_atten.number_of_edges()
edge_colors = range(2, M + 2)

coor_df = pd.DataFrame(adata.obsm['spatial'].copy(), index=adata.obs_names)
coor_df[1] = -1 * coor_df[1]
image_pos = dict(zip(range(coor_df.shape[0]), [np.array(coor_df.iloc[it,]) for it in range(coor_df.shape[0])]))

labels = nx.get_edge_attributes(G_atten,'weight')
plt.rcParams["figure.figsize"] = (3, 3)
fig, ax = plt.subplots(figsize=[10,9])
nx.draw_networkx_nodes(G_atten, image_pos, node_size=5, ax=ax)
cmap = plt.cm.plasma
edges = nx.draw_networkx_edges(G_atten, image_pos, edge_color=labels.values(),width=4, ax=ax,
                               edge_cmap=cmap,edge_vmax=0.25,edge_vmin=0.05)
ax = plt.gca()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = 0.05, vmax=0.25))
sm._A = []
plt.colorbar(sm)
plt.rcParams["figure.figsize"] = (3, 3)
ax.set_axis_off()
plt.axis('off')
plt.savefig(f'./outputs/{category}/attention0.5.jpg',
            bbox_inches='tight', dpi=200)
plt.show()
print(sm)





