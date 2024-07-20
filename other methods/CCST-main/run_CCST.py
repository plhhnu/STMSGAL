##exocrine GCNG with normalized graph matrix 
import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn import metrics
from scipy import sparse

import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader
from datetime import datetime 

rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath+'/ccst')


if __name__ == "__main__":
    import argparse
    import datetime
    now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    parser.add_argument( '--data_type', default='nsc', help='"sc" or "nsc", \
        refers to single cell resolution datasets(e.g. MERFISH) and \
        non single cell resolution data(e.g. ST) respectively') 
    # =========================== args ===============================
    parser.add_argument( '--data_name', type=str, default='', help="")
    parser.add_argument( '--lambda_I', type=float, default=0.3) #0.8 on MERFISH, 0.3 on ST
    parser.add_argument( '--data_path', type=str, default='generated_data/', help='data path')
    parser.add_argument( '--model_path', type=str, default='model') 
    parser.add_argument( '--embedding_data_path', type=str, default='Embedding_data') 
    parser.add_argument( '--result_path', type=str, default='results') 
    parser.add_argument( '--DGI', type=int, default=1, help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
    parser.add_argument( '--load', type=int, default=0, help='Load pretrained DGI model')
    parser.add_argument( '--loadembeding', type=int, default=0, help='Load embedding')

    parser.add_argument( '--num_epoch', type=int, default=5000, help='numebr of epoch in training DGI')
    parser.add_argument( '--hidden', type=int, default=256, help='hidden channels in DGI') 
    parser.add_argument( '--PCA', type=int, default=1, help='run PCA or not')   
    parser.add_argument( '--cluster', type=int, default=1, help='run cluster or not')
    parser.add_argument( '--n_clusters', type=int, default=5, help='number of clusters in Kmeans, when ground truth label is not avalible.') #5 on MERFISH, 20 on Breast
    parser.add_argument( '--draw_map', type=int, default=1, help='run drawing map')
    parser.add_argument( '--diff_gene', type=int, default=0, help='Run differential gene expression analysis')
    parser.add_argument('--datasets', type=str, default="Adult_Mouse_Brain (Coronal)", help='')
    parser.add_argument('--h5ad_path', type=str, default="h5ad", help='')
    parser.add_argument('--cluster_type', type=str, default="kmeans", help='')
    parser.add_argument('--parameter', type=str, default="", help='')
    args = parser.parse_args()


    print ('------------------------Model and Training Details--------------------------')
    print(args) 

    if args.data_type == 'sc': # should input a single cell resolution dataset, e.g. MERFISH
        from CCST_merfish_utils import CCST_on_MERFISH
        CCST_on_MERFISH(args)
    elif args.data_type == 'nsc': # should input a non-single cell resolution dataset, e.g. V1_Breast_Cancer_Block_A_Section_1
        from CCST_ST_utils import CCST_on_ST
        #'151507', '151508', '151509', '151510','151669',
        proj_list = [ '151507', '151508', '151509', '151510','151669',
                      '151670', '151671', '151672',
                     '151673',  '151674','151675', '151676']
        # proj_list = ['V1_Breast_Cancer_Block_A_Section_1']
        # proj_list = ['Adult Mouse Brain (FFPE)']
        ARI_list = []
        NMI_list = []
        parameter = ""
        for proj_idx in range(len(proj_list)):

            args.data_name = proj_list[proj_idx]
            print('===== Project ' + str(proj_idx + 1) + ' : ' + args.data_name)
            args.embedding_data_path = f'Embedding_data/{args.data_name}/'
            args.model_path = f'model/{args.data_name}/'
            args.result_path = f'results/{args.data_name}/'
            args.h5ad_path = f'h5ad/DLPFC/{args.data_name}'
            # args.h5ad_path = f'h5ad/{args.data_name}'
            if not os.path.exists(args.embedding_data_path):
                os.makedirs(args.embedding_data_path)
            if not os.path.exists(args.h5ad_path):
                os.makedirs(args.h5ad_path)
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            args.result_path = args.result_path + 'lambdaI' + str(args.lambda_I) + '/'
            if not os.path.exists(args.result_path):
                os.makedirs(args.result_path)
            CCST_on_ST(args,ARI_list, NMI_list, args.parameter)

        import CCST_ST_utils
        CCST_ST_utils.saveCsv(ARI_list, args.parameter, ARImean)
    else:
        print('Data type not specified')


    
