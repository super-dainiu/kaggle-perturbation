import os
import pickle

os.chdir('/home/yz979/code/kaggle-perturbation/')

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import requests
import scanpy as sc
import torch
import torch.nn as nn
from tqdm import tqdm

__all__ = [
    "GeneSimNetwork",
    "gene_sim_network",
]

def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    
    if os.path.exists(save_path):
        print('Found local copy...')
    else:
        print("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

def get_go_auto(gene_list, data_path):
    """
    Get gene ontology data

    Args:
        gene_list (list): list of gene names
        data_path (str): path to data

    Returns:
        pd.DataFrame: gene ontology data
    """
    
    data_path.mkdir(parents=True, exist_ok=True)
    go_path = os.path.join(data_path, 'go.csv')
    
    if os.path.exists(go_path):
        return pd.read_csv(go_path)
    else:
        if not os.path.exists(os.path.join(data_path, 'gene2go.pkl')):
            # download gene2go.pkl
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417' 
            dataverse_download(server_path, os.path.join(data_path, 'gene2go.pkl'))
            
        with open(os.path.join(data_path, 'gene2go.pkl'), 'rb') as f:
            gene2go = pickle.load(f)

        # Filter gene2go mapping to current genes
        gene2go = {i: list(gene2go[i]) for i in gene_list if i in gene2go}
        edge_list = []
        for g1 in tqdm(gene2go.keys()):
            for g2 in gene2go.keys():
                edge_list.append((g1, g2, len(np.intersect1d(gene2go[g1], gene2go[g2]))/
                                   len(np.union1d(gene2go[g1], gene2go[g2]))))

        # Filter edges
        edge_list = [i for i in edge_list if i[2] > 0.1]
        edge_df = pd.DataFrame(edge_list).rename(columns={0: 'gene1', 
                                                          1: 'gene2',
                                                          2: 'score'})

        edge_df = edge_df.rename(columns={'gene1': 'source',
                                          'gene2': 'target',
                                          'score': 'importance'})
        edge_df.to_csv(go_path, index=False)
        
        return edge_df


def get_coexpress_auto(adata, data_path, threshold=0.1):
    data_path.mkdir(parents=True, exist_ok=True)
    gc_path = os.path.join(data_path, 'coexpression.csv')

    if os.path.exists(gc_path):
        return pd.read_csv(gc_path)
    
    df = adata.to_df()
    gene_names = df.columns
    
    # Calculate correlation matrix
    cor_matrix = df.corr(method='pearson')

    # Filter edges
    edges = []
    for i in range(len(cor_matrix)):
        for j in range(i+1, len(cor_matrix)):
            if abs(cor_matrix.iloc[i, j]) > threshold:
                edges.append((gene_names[i], gene_names[j], cor_matrix.iloc[i, j]))
    
    edge_df = pd.DataFrame(edges, columns=['source', 'target', 'importance'])
    edge_df.to_csv(gc_path, index=False)
    return edge_df

def gene_sim_network(
    gene_list: List[str],
    node_map: Dict[str, int],
    network_type: str,
    adata: anndata.AnnData = None,
    data_path: str = './data',
    threshold: float = 0.1,
) -> "GeneSimNetwork":
    """
    Get gene similarity network

    Args:
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices
        network_type (str): type of network to use
        data_path (str): path to data
        threshold (float): threshold for coexpression network

    Returns:
        GeneSimNetwork: gene similarity network

    Usage:
        >>> gene_list = ['ENSG00000139618', 'ENSG00000141510', 'ENSG00000141510']
        >>> node_map = {'ENSG00000139618': 0, 'ENSG00000141510': 1}
        >>> network = get_gene_sim_network(gene_list, node_map, 'go')
    """
    data_path = Path(data_path)
    
    if network_type == 'go':
        edge_list = get_go_auto(gene_list, data_path)
    elif network_type == 'coexpression':
        edge_list = get_coexpress_auto(adata, data_path, threshold)

    network = GeneSimNetwork.from_edges(edge_list, gene_list, node_map)
    return network


@dataclass
class GeneSimNetwork():
    G: nx.DiGraph
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    
    @classmethod
    def from_edges(
        cls,
        edge_list: pd.DataFrame,
        gene_list: List,
        node_map: Dict[str, int],
        ) -> "GeneSimNetwork":
        """
        Generate gene similarity network from edge list

        Args:
            edge_list (pd.DataFrame): edge list of the network
            gene_list (list): list of gene names
            node_map (dict): dictionary mapping gene names to node indices

        Returns:
            GeneSimNetwork: gene similarity network
        """
        G = nx.from_pandas_edgelist(edge_list, source='source',
                                    target='target', edge_attr=['importance'],
                                    create_using=nx.DiGraph())
        for n in gene_list:
            if n not in G.nodes():
                G.add_node(n)

        to_remove = []
        for n in G.nodes():
            if n not in gene_list:
                to_remove.append(n)
        
        for n in to_remove:
            G.remove_node(n)

        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in G.edges]

        edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        
        edge_attr = nx.get_edge_attributes(G, 'importance') 
        importance = np.array([edge_attr[e] for e in G.edges])
        edge_weight = torch.Tensor(importance)
        
        return cls(G, edge_index, edge_weight)

if __name__ == '__main__':
    adata_path = 'data/adata_train.h5ad'
    de_path = 'data/de_train.h5ad'
    adata = anndata.read_h5ad(adata_path)
    de_train = anndata.read_h5ad(de_path)
    node_map = {i: j for i, j in zip(de_train.var, range(len(de_train.var)))}
    coexpress_network = gene_sim_network(adata.var, node_map, 'coexpression', adata=adata, data_path='data/grn')