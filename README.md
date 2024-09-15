# Unveiling Patterns in Spatial Transcriptomic Data: A Novel Approach Utilizing Graph Attention Autoencoder and Multi-Scale Subspace
## Requirements
tensorflow==2.5.0

pandas>=1.4.2

scipy>=1.4.1

scikit-learn==1.2.1

tqdm==4.64.1

matplotlib>=3.4.0

scanpy>=1.9.1

numpy>=1.19.2

numba>=0.56.4

seaborn==0.12.2

igraph==0.10.4

louvain==0.8.0

leidenalg==0.9.1

protobuf==3.20.3

pot==0.8.2

## Data

| Tissue                                        |                                                              |      |
| --------------------------------------------- | ------------------------------------------------------------ | ---- |
| Human dorsolateral pre-frontal cortex (DLPFC) | http://research.libd.org/spatialLIBD/                        |      |
| Adult Mouse Brain (FFPE)                      | https://www.10xgenomics.com/resources/datasets/adult-mouse-brain-ffpe-1-standard-1-3-0 |      |
| Ductal_Carcinoma_in_situ                      | https://www.10xgenomics.com/resources/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0 |      |
| V1_Breast_Cancer_Block_A_Section_1            | https://www.10xgenomics.com/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0 |      |
| Mouse visual cortex STARmap                   | https://www.dropbox.com/sh/f7ebheru1lbz91s/AADm6D54GSEFXB1feRy6OSASa/visual_1020/20180505_BY3_1kgenes?dl=0&subfolder_nav_tracking=1  |      |
| Mouse embryos of Stereo-seq                   | https://db.cngb.org/stomics/mosta/  |      |
## Usage
DLPFCtest.py: run STMSGAL on DLPFC

main.py: run STMSGAL on Adult Mouse Brain (FFPE) and Ductal_Carcinoma_in_situ

breatmain.py: run STMSGAL on V1_Breast_Cancer_Block_A_Section_1

