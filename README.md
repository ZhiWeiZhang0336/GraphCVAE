# GraphCVAE: Promoting Cell Heterogeneity and Therapeutic Target Discovery through Residual and Contrastive Learning
# Overview
GraphCVAE is a comprehensive model that integrates gene expression, spatial coordinates, and tissue morphology to accurately define spatial domains. The model first processes tissue image data to generate a morphological feature matrix, then combines spatial proximity information and gene expression similarity to enhance gene expression data. GraphCVAE employs multi-layer Graph Convolutional Networks (GCN) and a Variational Autoencoder (VAE) to optimize spatial domain representation. GCN captures complex spatial relationships, while VAE transforms high-dimensional data into a low-dimensional latent space. To improve model robustness, GraphCVAE adopts a contrastive learning approach, distinguishing between original and shuffled spatial gene expression data. The model accurately captures spatial patterns and gene expression features by combining reconstruction loss, KL divergence loss, and contrastive loss. The resulting embeddings are used for downstream analysis, including spatial domain identification, batch effect correction, and other spatial transcriptomics-related tasks, revealing tissue spatial structure and deepening understanding of gene expression patterns.
![Uploading image.png…](https://github.com/ZhiWeiZhang0336/GraphCVAE/blob/main/Overview/Workflow.png)

# Requirements
python == 3.9  
torch == 1.13.0  
scanpy == 1.9.2  
anndata == 0.8.0  
numpy == 1.22.3

# install
wget https://github.com/ZhiWeiZhang0336/GraphCVAE/archive/refs/heads/main.zip

unzip main.zip

cd /home/.../GraphCVAE  ### your own path

conda create -n GraphCVAE_env python=3.9

conda activate GraphCVAE_env

## step1 Installing PyTorch’s CUDA support or CPU support on Linux
pip3 install torch==1.13.0+cu116 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116  #### GPU

pip3 install torch==1.13.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu  #### CPU

## step2 Installing PyG package. If unsuccessful, refer to the "Install PyG package".
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html #### GPU

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html  ### CPU

## step3 Download other dependencies
pip install -r requirements.txt

# Dataset
1) DLPFC (Dorsolateral Prefrontal Cortex) Dataset: This dataset is accessible via the spatialLIBD platform at http://spatial.libd.org/spatialLIBD;
   
2) Human Brain Cancer, Human Colon Cancer, Mouse Brain Datasets: These datasets are available for download from the 10X Genomics website at https://www.10xgenomics.com/datasets ;
   
3) Mouse Embryo Dataset: This dataset can be accessed through the China National GeneBank's Stomics platform at https://db.cngb.org/stomics/mosta .
