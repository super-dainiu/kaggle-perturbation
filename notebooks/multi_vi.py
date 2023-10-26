#!/usr/bin/env python
# coding: utf-8

# # Integration of multi-modal data using MultiVI
# 
# MultiVI `[Ashuach et al., 2021]` is used to integrate multiomic datasets with single-modality (expression or accessibility) datasets. We are going to generate a latent embedding of the multiomic datasets for each cell type.

# ## Setting up environment

# In[ ]:


import os
os.chdir('/home/yz979/code/kaggle-perturbation')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# In[ ]:


import scanpy as sc
import numpy as np
import pandas as pd
import scvi
import scvi.model
import torch

torch.set_float32_matmul_precision('medium')


# ## Loading dataset
# 
# The preprocessed Multiome dataset with `h5ad` extension used for saving/loading anndata objects can be loaded using the `sc.read_h5ad` function. The datasets is not normalized or preprocessed.

# In[ ]:


data_dir = 'data'
adata_mvi = sc.read_h5ad(os.path.join(data_dir, 'multiome_train.h5ad'))
n_genes = sum(adata_mvi.var['feature_type'] == 'Gene Expression')
n_regions = sum(adata_mvi.var['feature_type'] == 'Peaks')
adata_mvi.X = adata_mvi.X.toarray()

print('n_genes: ', n_genes)
print('n_regions: ', n_regions)
print(adata_mvi)


# ## Dataset setup
# Now is the time to setup the dataset for MultiVI to prepare the dataset for training. Just like scvi-tools models, you can call `scvi.model.MULTIVI.setup_anndata` to setup your data. This function will accept the following arguments:
# 
# - `adata`: AnnData object containing the data to be preprocessed
# - `batch_key`: The key in `adata.obs` that contains the batch information
# - `categorical_covariate_keys`: A list of keys in `adata.obs` that contain categorical covariates
# - `continuous_covariate_keys`: A list of keys in `adata.obs` that contain continuous covariates
# - `protein_expression_obsm_key`: key in `adata.obsm` for protein expression data.
# - `protein_names_uns_key`: key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]` if it is a DataFrame, else will assign sequential names to proteins.

# In[4]:


scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key='donor_id')


# ## MultiVI Model
# 
# You can create a MultiVI model by creating an object from `scvi.model.MULTIVI` class. The constructor of this class takes the following arguments:
# 
# **Data related parameters:** 
# - `adata`: AnnData object containing train/valid/test data
# 
# **Model architecture parameters:**
# - `n_genes`: Number of genes in the expression data
# - `n_regions`: Number of regions in the accessibility data 
# - `n_latent`: Dimensionality of the latent space. If `None`, defaults to square root
#         of `n_hidden`.
# - `n_hidden`: Number of nodes per hidden layer. If `None`, defaults to square root
#         of number of regions.
# - `modality_weight`: Weighting scheme across modalities. One of the following:
#    - ``"equal"``: Equal weight in each modality
#    - ``"universal"``: Learn weights across modalities w_m.
#    - ``"cell"``: Learn weights across modalities and cells. w_{m,c}
# - `modality_penalty`: Training Penalty across modalities. One of the following:
#    - ``"Jeffreys"``: Jeffreys penalty to align modalities
#    - ``"MMD"``: MMD penalty to align modalities
#    - ``"None"``: No penalty
# - `dropout_rate`: Dropout rate for neural networks.
# - `latent_distribution`: Distribution of the latent space. One of the following:
#    - ``"normal"``: Normal distribution
#    - ``"ln"``: Log-normal distribution
# - `use_batch_norm`: Whether to use batch norm in the neural networks. Only of the following:
#    - ``"none"``: No batch norm
#    - ``"encoder"``: Batch norm in the encoder
#    - ``"decoder"``: Batch norm in the decoder
#    - ``"both"``: Batch norm in both encoder and decoder
# - `use_layer_norm`: Whether to use layer norm in the neural networks. Only of the following:
#    - ``"none"``: No layer norm
#    - ``"encoder"``: Layer norm in the encoder
#    - ``"decoder"``: Layer norm in the decoder
#    - ``"both"``: Layer norm in both encoder and decoder

# In[5]:


model_params = dict(
    n_genes = n_genes,
    n_regions = n_regions,
    n_latent = 64,
    n_hidden = 512,
    modality_weights = 'cell',
    modality_penalty = 'Jeffreys',
    gene_likelihood = 'zinb',
    use_batch_norm = 'none',
    use_layer_norm = 'both',
    latent_distribution = 'normal',
    dropout_rate = 0.1,
)
mvi = scvi.model.MULTIVI(adata_mvi, **model_params)


# ## Training MultiVI
# 
# In order to train your MultiVI model, you need to use `train` function of your `model`. This function accepts the following parameters:
# - `max_epochs`: Maximum number of epochs to train the model. CPA generally converges after high number of epochs, so you can set this to a high value.
# - `use_gpu`: If you have a GPU, you can set this to `True` to speed up the training process.
# - `batch_size`: Batch size for training. You can set this to a high value (e.g. 512, 1024, 2048) if you have a GPU. 
# - `lr`: Learning rate for training.
# - `weight_decay`: Weight decay for training.
# - `early_stopping`: Whether to use early stopping or not.
# - `early_stopping_patience`: Number of epochs to wait before stopping training if validation metric does not improve.
# - `check_val_every_n_epoch`: Number of epochs to wait before running validation.
# 

# In[10]:


train_params = dict(
    max_epochs = 500,
    lr = 1e-3,
    use_gpu = True,
    batch_size = 8,
    weight_decay = 1e-5,
    check_val_every_n_epoch = 1,
    early_stopping = True,
    save_best = True,
)
mvi.train(**train_params)
mvi.save('models/mvi', overwrite=True)


# In[ ]:


mvi: scvi.model.MULTIVI = scvi.model.MULTIVI.load('models/mvi', adata_mvi)


# In[ ]:


latent_outputs = mvi.get_latent_representation()

