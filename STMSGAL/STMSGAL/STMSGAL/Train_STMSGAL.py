import numpy as np
import scipy.sparse as sp
from .STMSGAL import STMSGAL
#import tensorflow.compat.v1 as tf
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
import pandas as pd
import scanpy as sc
import random
from natsort import natsorted


random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
def train_STMSGAL(adata, hidden_dims=[512,30], alpha=0, n_epochs=100, lr=0.0001, key_added='STMSGAL',
                gradient_clipping=5, nonlinear=True, weight_decay=0.0001,verbose=True, 
                random_seed=2020, pre_labels=None, pre_resolution=0.2,pca = False,
                save_attention=False, save_loss=False, save_reconstrction=False,
                  n_cluster = 6,reg_ssc_coef=1.0,cost_ssc_coef=1.0,
                  L_self_supervised_coef = 1.0,category = "dataset"):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    alpha
        The weight of cell type-aware spatial neighbor network.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    nonlinear
        If True, the nonlinear avtivation is performed.
    weight_decay
        Weight decay for AdamOptimizer.
    pre_labels
        The key in adata.obs for the manually designate the pre-clustering results. Only used when alpha>0.
    pre_resolution
        The resolution parameter of sc.tl.louvain for the pre-clustering. Only used when alpha>0 and per_labels==None.
    save_attention
        If True, the weights of the attention layers are saved in adata.uns['STMSGAL_attention']
    save_loss
        If True, the training loss is saved in adata.uns['STMSGAL_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STMSGAL_ReX'].

    Returns
    -------
    AnnData
    """

    tf.reset_default_graph()
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    X = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G_tf = prepare_graph_data(G)

    batchsize = cells.shape[0]
    trainer = STMSGAL(hidden_dims=[X.shape[1]] + hidden_dims, alpha=alpha,
                    n_epochs=n_epochs, lr=lr, gradient_clipping=gradient_clipping,
                    nonlinear=nonlinear,weight_decay=weight_decay, verbose=verbose,
                    random_seed=random_seed,batchsize = batchsize,n_cluster = n_cluster,
                      reg_ssc_coef = reg_ssc_coef,cost_ssc_coef = cost_ssc_coef,
                      L_self_supervised_coef = L_self_supervised_coef,category=category)
    if alpha == 0:
        trainer(G_tf, G_tf, X)
        embeddings, attentions, loss, ReX, pred_dsc= trainer.infer(G_tf, G_tf, X)
    else:
        G_df = Spatial_Net.copy()
        if pre_labels==None:
            if verbose:
                print('------Pre-clustering using louvain with resolution=%.2f' %pre_resolution)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata)
            sc.tl.louvain(adata, resolution=pre_resolution, key_added='expression_louvain_label')
            pre_labels = 'expression_louvain_label'

        prune_G_df = prune_spatial_Net(G_df, adata.obs[pre_labels])
        prune_G_df['Cell1'] = prune_G_df['Cell1'].map(cells_id_tran)
        prune_G_df['Cell2'] = prune_G_df['Cell2'].map(cells_id_tran)
        prune_G = sp.coo_matrix((np.ones(prune_G_df.shape[0]), (prune_G_df['Cell1'], prune_G_df['Cell2'])))
        prune_G_tf = prepare_graph_data(prune_G)
        prune_G_tf = (prune_G_tf[0], prune_G_tf[1], G_tf[2])
        trainer(G_tf, prune_G_tf, X)
        embeddings, attentions, loss, ReX ,pred_dsc= trainer.infer(G_tf, prune_G_tf, X)
    if pca:
        from sklearn.decomposition import PCA
        print('Shape of data to PCA:', X.shape)
        pca = PCA(n_components=30)
        X_PC = pca.fit_transform(X)
        print('Shape of data output by PCA:', X_PC.shape)
        print('PCA recover:', pca.explained_variance_ratio_.sum())
        embeddings = X_PC
    cell_reps = pd.DataFrame(embeddings)
    cell_reps.index = cells

    adata.obsm[key_added] = cell_reps.loc[adata.obs_names, ].values
    if save_attention:
        adata.uns['STMSGAL_attention'] = attentions[0].toarray()
    if save_loss:
        adata.uns['STMSGAL_loss'] = loss
    if save_reconstrction:
        ReX = pd.DataFrame(ReX, index=X.index, columns=X.columns)
        ReX[ReX<0] = 0
        #adata.layers['STMSGAL_ReX'] = ReX.values
        adata.obsm['STMSGAL_ReX'] = ReX.values
    return adata,pred_dsc


def prune_spatial_Net(Graph_df, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' %Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label']==Graph_df['Cell2_label'],]
    print('%d edges after pruning.' %Graph_df.shape[0])
    return Graph_df


def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)# self-loop
    #data =  adj.tocoo().data
    #adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape)

def recovery_Imputed_Count(adata, size_factor):
    assert('ReX' in adata.uns)
    temp_df = adata.uns['ReX'].copy()
    sf = size_factor.loc[temp_df.index]
    temp_df = np.expm1(temp_df)
    temp_df = (temp_df.T * sf).T
    adata.uns['ReX_Count'] = temp_df
    return adata
