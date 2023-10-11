#!/usr/bin/env python
# coding: utf-8

# # Predicting perturbation responses for unseen cell-types (context transfer)

# In this tutorial, we will train and evaluate a CPA model on the preprocessed Kang PBMC dataset (See Sup Figures 2-3 [here](https://www.embopress.org/action/downloadSupplement?doi=10.15252%2Fmsb.202211517&file=msb202211517-sup-0001-Appendix.pdf) for a deeper dive).
# 
# The following steps are going to be covered:
# 1. Setting up environment
# 2. Loading the dataset
# 3. Preprocessing the dataset
# 4. Creating a CPA model
# 5. Training the model
# 6. Latent space visualisation
# 7. Prediction evaluation across different perturbations

# ## Setting up environment

# In[1]:


import sys
#if branch is stable, will install via pypi, else will install from source
branch = "latest"
IN_COLAB = "google.colab" in sys.modules

# In[1]:

import os
os.chdir('/home/yz979/code/kaggle-perturbation')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# In[2]:

import cpa
import scanpy as sc
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

torch.set_float32_matmul_precision('medium')


# In[3]:


sc.settings.set_figure_params(dpi=100)


# In[4]:


train_data_path = 'data/adata_train.h5ad'


# ## Loading dataset
# 
# The preprocessed Kang PBMC dataset with `h5ad` extension used for saving/loading anndata objects is publicly available in the [Google Drive](https://drive.google.com/drive/u/0/folders/1yFB0gBr72_KLLp1asojxTgTqgz6cwpju) and can be loaded using the `sc.read` function with the `backup_url` argument. The datasets is normalized & pre-processed using `scanpy`. Top 5000 highly variable genes are selected. 

# In[5]:


adata = sc.read(train_data_path)
adata

# # inverse log1p
# print(adata.X.min(), adata.X.max())
# adata.X = np.expm1(adata.X)
# print(adata.X.min(), adata.X.max())
print(adata.obs['cell_type'].value_counts())

# rename columns in adata.obs
adata.obs.rename(columns={'sm_lincs_id': 'condition'}, inplace=True)
print(adata.obs['condition'].value_counts())

# obs['control'] to true and false (lowercase)
adata.obs['control'] = adata.obs['control'].astype(str).map(lambda x: x.lower())
print(adata.obs['control'].value_counts())

print(adata.obs['dose_uM'].value_counts())


# In[9]:


adata.obs.keys()


# ## Dataset setup
# Now is the time to setup the dataset for CPA to prepare the dataset for training. Just like scvi-tools models, you can call `cpa.CPA.setup_anndata` to setup your data. This function will accept the following arguments:
# 
# - `adata`: AnnData object containing the data to be preprocessed
# - `perturbation_key`: The key in `adata.obs` that contains the perturbation information
# - `control_group`: The name of the control group in `perturbation_key`
# - `batch_key`: The key in `adata.obs` that contains the batch information
# - `dosage_key`: The key in `adata.obs` that contains the dosage information
# - `categorical_covariate_keys`: A list of keys in `adata.obs` that contain categorical covariates
# - `is_count_data`: Whether the `adata.X` is count data or not
# - `deg_uns_key`: The key in `adata.uns` that contains the differential expression results
# - `deg_uns_cat_key`: The key in `adata.obs` that contains the category information of each cell which can be used as to access differential expression results in `adata.uns[deg_uns_key]`. For example, if `deg_uns_key` is `rank_genes_groups_cov` and `deg_uns_cat_key` is `cov_cond`, then `adata.uns[deg_uns_key][cov_cond]` will contain the differential expression results for each category in `cov_cond`.
# - `max_comb_len`: The maximum number of perturbations that are applied to each cell. For example, if `max_comb_len` is 2, then the model will be trained to predict the effect of single perturbations and the effect of double perturbations.

# We will create a dummy dosage variable for each condition (control, IFN-beta stimulated). It is recommended to use Identity (i.e. doser_type = 'identity') for dosage scaling function when there is no dosage information available.

# In[27]:


adata.obs['cell_type'].value_counts()


# In[8]:


adata.obs['condition'].value_counts()


# In[6]:


cpa.CPA.setup_anndata(adata, 
                      perturbation_key='condition',
                      control_group='control',
                      dosage_key='dose_uM',
                      categorical_covariate_keys=['cell_type', 'library_id', 'plate_name', 'well', 'donor_id'],
                      is_count_data=False,
                      max_comb_len=1,
                     )


# In[8]:


model_params = {
    "n_latent": 256,
    "recon_loss": "gauss",
    "doser_type": "linear",
    "n_hidden_encoder": 1024,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 2048,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": True,
    "dropout_rate_encoder": 0.2,
    "dropout_rate_decoder": 0.2,
    "variational": True,
    "seed": 6977,
}

trainer_params = {
    "n_epochs_kl_warmup": 5,
    "n_epochs_pretrain_ae": 5,
    "n_epochs_adv_warmup": 5,
    "n_epochs_mixup_warmup": 0,
    "n_epochs_verbose": 1,
    "mixup_alpha": 0.0,
    "adv_steps": 5,
    "n_hidden_adv": 64,
    "n_layers_adv": 3,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 5.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,
    "step_size_lr": 10,
}


# ## CPA Model
# 
# You can create a CPA model by creating an object from `cpa.CPA` class. The constructor of this class takes the following arguments:
# **Data related parameters:** 
# - `adata`: AnnData object containing train/valid/test data
# - `split_key`: The key in `adata.obs` that contains the split information
# - `train_split`: The value in `split_key` that corresponds to the training data
# - `valid_split`: The value in `split_key` that corresponds to the validation data
# - `test_split`: The value in `split_key` that corresponds to the test data
# **Model architecture parameters:**
# - `n_latent`: Number of latent dimensions
# - `recon_loss`: Reconstruction loss function. Currently, Supported losses are `nb`, `zinb`, and `gauss`.
# - `n_hidden_encoder`: Number of hidden units in the encoder
# - `n_layers_encoder`: Number of layers in the encoder
# - `n_hidden_decoder`: Number of hidden units in the decoder
# - `n_layers_decoder`: Number of layers in the decoder
# - `use_batch_norm_encoder`: Whether to use batch normalization in the encoder
# - `use_layer_norm_encoder`: Whether to use layer normalization in the encoder
# - `use_batch_norm_decoder`: Whether to use batch normalization in the decoder
# - `use_layer_norm_decoder`: Whether to use layer normalization in the decoder
# - `dropout_rate_encoder`: Dropout rate in the encoder
# - `dropout_rate_decoder`: Dropout rate in the decoder
# - `variational`: Whether to use variational inference. NOTE: False is highly recommended.
# - `seed`: Random seed

# In this notebook, we left out `B` cells treated with  `IFN-beta` from the training dataset (OOD set) and randomly split the remaining cells into train/valid sets. The split information is stored in `adata.obs['split_B']` column. We would like to see if the model can predict how `B` cells can respond to `IFN-beta` stimulation.

# In[9]:


model = cpa.CPA(adata=adata, 
                **model_params,
               )


# ## Training CPA
# 
# In order to train your CPA model, you need to use `train` function of your `model`. This function accepts the following parameters:
# - `max_epochs`: Maximum number of epochs to train the model. CPA generally converges after high number of epochs, so you can set this to a high value.
# - `use_gpu`: If you have a GPU, you can set this to `True` to speed up the training process.
# - `batch_size`: Batch size for training. You can set this to a high value (e.g. 512, 1024, 2048) if you have a GPU. 
# - `plan_kwargs`: dictionary of parameters passed the CPA's `TrainingPlan`. You can set the following parameters:
#     * `n_epochs_adv_warmup`: Number of epochs to linearly increase the weight of adversarial loss. 
#     * `n_epochs_mixup_warmup`: Number of epochs to linearly increase the weight of mixup loss.
#     * `n_epochs_pretrain_ae`: Number of epochs to pretrain the autoencoder.
#     * `lr`: Learning rate for training autoencoder.
#     * `wd`: Weight decay for training autoencoder.
#     * `adv_lr`: Learning rate for training adversary.
#     * `adv_wd`: Weight decay for training adversary.
#     * `adv_steps`: Number of steps to train adversary for each step of autoencoder.
#     * `reg_adv`: Maximum Weight of adversarial loss.
#     * `pen_adv`: Penalty weight of adversarial loss.
#     * `n_layers_adv`: Number of layers in adversary.
#     * `n_hidden_adv`: Number of hidden units in adversary.
#     * `use_batch_norm_adv`: Whether to use batch normalization in adversary.
#     * `use_layer_norm_adv`: Whether to use layer normalization in adversary.
#     * `dropout_rate_adv`: Dropout rate in adversary.
#     * `step_size_lr`: Step size for learning rate scheduler.
#     * `do_clip_grad`: Whether to clip gradients by norm.
#     * `clip_grad_value`: Maximum value of gradient norm.
#     * `adv_loss`: Type of adversarial loss. Can be either `cce` for Cross Entropy loss or `focal` for Focal loss.
#     * `n_epochs_verbose`: Number of epochs to print latent information disentanglement evaluation.
# - `early_stopping_patience`: Number of epochs to wait before stopping training if validation metric does not improve.
# - `check_val_every_n_epoch`: Number of epochs to wait before running validation.
# - `save_path`: Path to save the best model after training.
# 
# 

# In[10]:


model.train(max_epochs=50,
            use_gpu=True, 
            batch_size=4096,
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=1,
            save_path='data/cpa_model_adata',
           )


# ## Restore best model
# 
# In case you have already saved your pretrained model, you can restore it using the following code. The `cpa.CPA.load` function accepts the following arguments:
# - `dir_path`: path to the directory where the model is saved
# - `adata`: anndata object
# - `use_gpu`: whether to use GPU or not
# 

# In[ ]:


model: cpa.CPA = cpa.CPA.load(dir_path='data/cpa_model_2',
                     adata=adata,
                     use_gpu=True)


# ## Latent Space Visualization
# 
# latent vectors of all cells can be computed with `get_latent_representation` function. This function produces a python dictionary with the following keys:
# - `latent_basal`: latent vectors of all cells in basal state of autoencoder
# - `latent_after`: final latent vectors which can be used for decoding
# - `latent_corrected`: batch-corrected latents if batch_key was provided

# In[ ]:


latent_outputs = model.get_latent_representation(adata, batch_size=2048)


# In[ ]:


sc.pp.neighbors(latent_outputs['latent_basal'])
sc.tl.umap(latent_outputs['latent_basal'])


# As observed below, the basal representation should be free of the variation(s) of the `condition` and `cell_type`. 

# In[ ]:


sc.pl.umap(latent_outputs['latent_basal'],
           color=['cell_type', 'condition', 'donor_id', 'library_id', 'plate_name', 'well'],
           frameon=False, 
           wspace=0.3,
           save='_basal.png',
           show=False)


# In[ ]:


sc.pp.neighbors(latent_outputs['latent_after'])
sc.tl.umap(latent_outputs['latent_after'])


# Here, you can visualize that when the `condition` and `cell_type` embeddings are added to the basal representation,
# As you can see now cell types and conditions are separated. 

# In[ ]:


sc.pl.umap(latent_outputs['latent_after'], 
           color=['cell_type', 'condition', 'donor_id', 'library_id', 'plate_name', 'well'],
           frameon=False,
           wspace=0.3,
           save='_after.png',
           show=False)


# ## Evaluation
# 

# To evaluate the model's prediction performance, we can use `model.predict()` function. $R^2$ score for each combination of `<cell_type, stimulated>` is computed over mean statistics of the top 50, 20, and 10 DEGs (including all genes). CPA transfers the context from control to IFN-beta stimulated for each cell type. Next, we will evaluate the model's prediction performance on the whole dataset, including OOD (test) cells. The model will report metrics on how well we have
# captured the variation in top `n` differentially expressed genes when compared to control cells
# (`CTRL`)  for each condition. The metrics calculate the mean accuracy (`r2_mean_deg`), the variance (`r2_var_deg`) and similar metrics (`r2_mean_lfc_deg` and `log fold change`)to measure the log fold change of the predicted cells vs control`((LFC(control, ground truth) ~ LFC(control, predicted cells))`.  The `R2` is the `sklearn.metrics.r2_score` from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

# In[ ]:


model.predict(adata, batch_size=128)

CPA_pred = adata.obsm['CPA_pred']
print(mean_squared_error(adata.X, CPA_pred, squared=False))

