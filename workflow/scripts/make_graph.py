#!/usr/bin/env python3
"""
Graph Construction Script for Drug Response Prediction

This script processes biological interaction networks, drug-target interactions,
and gene expression data to create a heterogeneous graph for drug response prediction.

The resulting graph contains:
- Drug nodes connected to their protein targets
- Protein-protein interactions (from multiple databases)
- Transcription factor -> RNA interactions
- RNA -> Protein translation edges
- Protein -> Pathway membership edges
- Expression input nodes for each RNA
- Output node for drug response (AUC)
"""

import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
from pathlib import Path
from pypath.utils import mapping
import joblib
from scipy import stats

from gsnn.simulate.nx2pyg import nx2pyg
from lincs_gsnn.proc.get_bio_interactions import get_bio_interactions
from lincs_gsnn.proc.subset import subset_graph


class ExpressionNormalizer:
    def __init__(self, method, quantile_clip=(0.02, 0.98), n_top_variable_genes=None, var_quantile=None):
        self.method = method
        self.quantile_clip = quantile_clip
        self.n_top_variable_genes = n_top_variable_genes
        self.var_quantile = var_quantile
        self.params = {}
        self.fitted_genes = None  # Store gene names/order from fitting
        self.selected_genes = None  # Store genes selected by variance filtering

    def fit(self, df):
        # Apply variance-based gene selection if specified
        if self.n_top_variable_genes is not None or self.var_quantile is not None:
            df = self._select_variable_genes(df)
        
        # Store the genes we fitted on for consistent transformation
        self.fitted_genes = df.columns.tolist()
        
        if self.method == 'unit_norm':
            self.params['min'] = df.min(axis=0)
            self.params['max'] = df.max(axis=0)
        elif self.method == 'zscore':
            self.params['mean'] = df.mean(axis=0)
            self.params['std'] = df.std(axis=0)
        elif self.method == 'robust_z':
            self.params['median'] = df.median(axis=0)
            self.params['mad'] = (df - df.median(axis=0)).abs().median(axis=0)
        elif self.method in ['winsorized_z', 'winsorized_unit_norm']:
            q_low, q_high = self.quantile_clip
            self.params['q_low'] = df.quantile(q_low, axis=0)
            self.params['q_high'] = df.quantile(q_high, axis=0)
            df_clip = df.clip(lower=self.params['q_low'], upper=self.params['q_high'], axis=1)
            if self.method == 'winsorized_z':
                self.params['mean'] = df_clip.mean(axis=0)
                self.params['std'] = df_clip.std(axis=0)
            else: # winsorized_unit_norm
                self.params['min'] = df_clip.min(axis=0)
                self.params['max'] = df_clip.max(axis=0)
        elif self.method == 'quantile':
            # Store the reference distribution (mean across samples for each quantile)
            # Each gene's values are ranked and mapped to the same reference distribution
            self.params['reference_dist'] = df.mean(axis=1).sort_values()
        elif self.method == 'rank_inv_normal':
            # For rank-based inverse normal, we don't need to store parameters
            # Each gene is independently transformed to N(0,1) based on its ranks
            pass
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def _align_dataframe(self, df):
        """Align dataframe to match fitted genes, handling missing/extra genes."""
        if self.fitted_genes is None:
            raise ValueError("Normalizer must be fitted before transforming data")
        
        # First filter to selected genes if variance filtering was applied
        if self.selected_genes is not None:
            df = df[[gene for gene in self.selected_genes if gene in df.columns]]
        
        # Drop extra genes not in fitted data
        available_genes = [gene for gene in self.fitted_genes if gene in df.columns]
        missing_genes = [gene for gene in self.fitted_genes if gene not in df.columns]
        
        if missing_genes:
            print(f"Warning: {len(missing_genes)} genes missing from new data, will use mean imputation")
        
        # Select only available genes in correct order
        df_aligned = df[available_genes].copy()
        
        # Add missing genes with mean imputation (use fitted means if available, otherwise 0)
        for gene in missing_genes:
            if self.method in ['zscore', 'winsorized_z'] and 'mean' in self.params:
                impute_value = self.params['mean'][gene]
            elif self.method == 'robust_z' and 'median' in self.params:
                impute_value = self.params['median'][gene]
            elif self.method in ['unit_norm', 'winsorized_unit_norm'] and 'min' in self.params:
                # For unit norm, use midpoint of min-max range
                impute_value = (self.params['min'][gene] + self.params['max'][gene]) / 2
            elif self.method == 'quantile':
                # For quantile normalization, use median of reference distribution
                impute_value = self.params['reference_dist'].median()
            elif self.method == 'rank_inv_normal':
                # For rank inverse normal, use 0 (will become 0 after transformation)
                impute_value = 0.0
            else:
                impute_value = 0.0  # Fallback
            
            df_aligned[gene] = impute_value
        
        # Reorder to match fitted gene order
        df_aligned = df_aligned[self.fitted_genes]
        
        return df_aligned

    def _select_variable_genes(self, df):
        """Select top variable genes based on variance criteria."""
        print("Selecting variable genes...")
        
        # Calculate variance for each gene (column)
        gene_vars = df.var(axis=0)
        
        if self.n_top_variable_genes is not None:
            # Select top N most variable genes
            selected_genes = gene_vars.nlargest(self.n_top_variable_genes).index.tolist()
            print(f"Selected top {self.n_top_variable_genes} most variable genes")
        elif self.var_quantile is not None:
            # Select genes above variance quantile threshold
            var_threshold = np.quantile(gene_vars, self.var_quantile)
            selected_genes = gene_vars[gene_vars > var_threshold].index.tolist()
            print(f"Selected {len(selected_genes)} genes above {self.var_quantile:.2f} variance quantile")
        else:
            selected_genes = df.columns.tolist()
        
        # Store selected genes for future reference
        self.selected_genes = selected_genes
        
        # Return dataframe with only selected genes
        return df[selected_genes]

    def _quantile_normalize(self, df):
        """Apply quantile normalization using the fitted reference distribution."""
        df_transformed = df.copy()
        reference_dist = self.params['reference_dist'].values
        
        for gene in df.columns:
            # Rank the values for this gene
            ranks = df[gene].rank(method='average')
            # Convert ranks to quantiles (0 to 1)
            quantiles = (ranks - 1) / (len(ranks) - 1)
            # Map quantiles to reference distribution
            indices = (quantiles * (len(reference_dist) - 1)).round().astype(int)
            # Clip indices to valid range
            indices = np.clip(indices, 0, len(reference_dist) - 1)
            df_transformed[gene] = reference_dist[indices]
            
        return df_transformed

    def _rank_inv_normal_transform(self, df):
        """Apply rank-based inverse normal (non-paranormal) transformation."""
        df_transformed = df.copy()
        
        for gene in df.columns:
            # Get ranks (using average method for ties)
            ranks = df[gene].rank(method='average')
            # Convert to quantiles (0 to 1, exclusive of 0 and 1)
            n = len(ranks)
            quantiles = (ranks - 0.5) / n
            # Apply inverse normal transformation
            df_transformed[gene] = stats.norm.ppf(quantiles)
            
        return df_transformed

    def transform(self, df):
        # Align dataframe to match fitted genes
        df_aligned = self._align_dataframe(df)
        
        if self.method == 'unit_norm':
            df_transformed = df_aligned - self.params['min']
            df_transformed = df_transformed / (self.params['max'] - self.params['min'])
        elif self.method == 'zscore':
            df_transformed = (df_aligned - self.params['mean']) / self.params['std']
        elif self.method == 'robust_z':
            df_transformed = (df_aligned - self.params['median']) / self.params['mad']
        elif self.method in ['winsorized_z', 'winsorized_unit_norm']:
            df_transformed = df_aligned.clip(lower=self.params['q_low'], upper=self.params['q_high'], axis=1)
            if self.method == 'winsorized_z':
                df_transformed = (df_transformed - self.params['mean']) / self.params['std']
            else:
                df_transformed = df_transformed - self.params['min']
                df_transformed = df_transformed / (self.params['max'] - self.params['min'])
        elif self.method == 'quantile':
            df_transformed = self._quantile_normalize(df_aligned)
        elif self.method == 'rank_inv_normal':
            df_transformed = self._rank_inv_normal_transform(df_aligned)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return df_transformed

    def save(self, path):
        joblib.dump({
            'method': self.method, 
            'quantile_clip': self.quantile_clip, 
            'n_top_variable_genes': self.n_top_variable_genes,
            'var_quantile': self.var_quantile,
            'params': self.params,
            'fitted_genes': self.fitted_genes,
            'selected_genes': self.selected_genes
        }, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        obj = cls(
            d['method'], 
            d.get('quantile_clip', (0.02, 0.98)),
            d.get('n_top_variable_genes', None),
            d.get('var_quantile', None)
        )
        obj.params = d['params']
        obj.fitted_genes = d.get('fitted_genes', None)
        obj.selected_genes = d.get('selected_genes', None)
        return obj


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create biological graph for drug response prediction')
    
    # Data paths
    parser.add_argument('--drug-data', type=str, 
                       default='/home/teddy/local/AMLVAE/data/beataml_probit_curve_fits_v4_distr.txt',
                       help='Path to drug response data')
    parser.add_argument('--clinical-data', type=str,
                       default='/home/teddy/local/AMLVAE/data/beataml_clinical_for_inputs.csv',
                       help='Path to clinical mapping data')
    parser.add_argument('--targetome-data', type=str,
                       default='/home/teddy/local/data/targetome_extended-01-23-25.csv',
                       help='Path to drug-target interaction data')
    parser.add_argument('--drug-meta', type=str,
                       default='../../data/beataml_drugs_for_targetome.csv',
                       help='Path to drug metadata with InChI keys')
    parser.add_argument('--reactome-data', type=str,
                       default='../../data/UniProt2Reactome.txt',
                       help='Path to Reactome pathway annotations')
    parser.add_argument('--expression-data', type=str,
                       default='/home/teddy/local/AMLVAE/data/aml_full_manuscript.csv',
                       help='Path to gene expression data')
    
    # Processing parameters
    parser.add_argument('--resp-norm', type=str, default='zscore',
                       choices=['zscore', 'unit_norm', 'within_drug_zscore', 'none'],
                       help='Response normalization method')
    parser.add_argument('--graph-depth', type=int, default=5,
                       help='Maximum depth for graph subsetting')
    parser.add_argument('--expr-n-top-genes', type=int, default=None,
                       help='Select top N most variable genes (takes precedence over --expr-var-threshold)')
    parser.add_argument('--expr-var-threshold', type=float, default=None,
                       help='Select genes above this variance quantile threshold (0.0-1.0)')
    parser.add_argument('--expr-norm', type=str, default='unit_norm',
                       choices=['unit_norm', 'zscore', 'robust_z', 'winsorized_z', 'winsorized_unit_norm', 'quantile', 'rank_inv_normal'],
                       help='Expression normalization method (unit_norm: 0-1 scaling, zscore: z-score, robust_z: (x-median)/MAD, winsorized_z/unit_norm: clip to quantiles then normalize, quantile: quantile normalization, rank_inv_normal: rank-based inverse normal transform)')
    parser.add_argument('--expr-clip-quantiles', type=float, nargs=2, default=[0.02, 0.98],
                       help='Quantiles for winsorized normalization (e.g. 0.02 0.98 for 2-98% clipping)')
    parser.add_argument('--expr-normalizer-save-path', type=str, default=None,
                       help='Path to save fitted expression normalizer (for future transformation)')
    parser.add_argument('--min-pathway-size', type=int, default=25,
                       help='Minimum number of proteins per pathway')
    parser.add_argument('--max-assay-value', type=float, default=1000.0,
                       help='Maximum Ki/Kd value for drug-target interactions')
    parser.add_argument('--include-mirna', type=lambda x: (str(x).lower() in ['true','1','yes','t']), default=True,
                      help='Include miRNA nodes/edges in biological graph (default: True)')
    parser.add_argument('--include-extra', type=lambda x: (str(x).lower() in ['true','1','yes','t']), default=True,
                      help='Include extra biological edges (default: True)')
    
    # Train/val split control (test set uses MDS logic)
    parser.add_argument('--train-frac', type=float, default=0.85,
                       help='Fraction of non-MDS samples for training (remainder goes to validation)')
    
    # Note: Test set assignment uses BeatAML-specific logic:
    # - Prior MDS patients are automatically assigned to test set
    # - Remaining samples split between train/val based on --train-frac
    
    # Output paths
    parser.add_argument('--output-dir', type=str, default='../proc',
                       help='Output directory for processed data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_and_process_drug_data(drug_path, clinical_path, resp_norm):
    """Load and process drug response data."""
    print("Loading drug response data...")
    
    # Load drug response data
    drug = pd.read_csv(drug_path, sep='\t')
    
    # Load clinical mapping
    aml_clin = pd.read_csv(clinical_path)[['gdc_id', 'labId']]
    aml_clin = aml_clin.rename(columns={'labId': 'lab_id', 'gdc_id': 'id'})
    
    # Merge and clean
    drug = drug.merge(aml_clin, on='lab_id', how='left')
    drug = drug[['id', 'inhibitor', 'auc']].dropna()
    
    # Split inhibitor column for combination drugs
    drug['inhibitor_1'] = drug.inhibitor.str.split(' - ').str[0]
    drug['inhibitor_2'] = drug.inhibitor.str.split(' - ').str[1]
    
    # Apply response normalization
    if resp_norm == 'zscore':
        drug['response'] = (drug.auc - drug.auc.mean()) / drug.auc.std()
    elif resp_norm == 'unit_norm':
        drug['response'] = (drug.auc - drug.auc.min()) / (drug.auc.max() - drug.auc.min())
    elif resp_norm == 'within_drug_zscore':
        drug['response'] = drug.groupby('inhibitor')['auc'].transform(lambda x: (x - x.mean()) / x.std())
    elif resp_norm == 'none':
        drug['response'] = drug['auc']
    else:
        raise ValueError(f'Invalid response normalization: {resp_norm}')
    
    print(f"Loaded {len(drug)} drug response measurements")
    return drug


def load_drug_target_interactions(targetome_path, drug_meta_path, drug_candidates, max_assay_value):
    """Load and filter drug-target interaction data."""
    print("Loading drug-target interactions...")
    
    # Load targetome data
    dtis = pd.read_csv(targetome_path, low_memory=False)
    
    # Filter for high-quality interactions
    dtis = dtis[
        (dtis.assay_type.isin(['Ki', 'Kd'])) & 
        (dtis.assay_relation.isin(['=', '<', '<='])) & 
        (dtis.assay_value < max_assay_value)
    ]
    
    # Load drug metadata and merge
    dti_meta = pd.read_csv(drug_meta_path)[['inhibitor', 'inchi_key']]
    dtis = dtis.merge(dti_meta, on='inchi_key', how='inner')
    
    # Filter for drugs in our dataset
    dtis = dtis[dtis.inhibitor.str.lower().isin(drug_candidates)]
    
    print(f"Loaded {len(dtis)} drug-target interactions for {len(dtis.inhibitor.unique())} drugs")
    return dtis


def create_dti_graph(dtis):
    """Create drug-target interaction edges."""
    dti_df = dtis.assign(
        source=['DRUG__' + x for x in dtis.inhibitor.str.lower()],
        target=['PROTEIN__' + x for x in dtis.uniprot_id],
        edge_type='dti'
    )[['source', 'target', 'edge_type']].drop_duplicates()
    
    return dti_df


def load_pathway_data(reactome_path, protspace, min_pathway_size):
    """Load and filter pathway annotation data."""
    print("Loading pathway annotations...")
    
    pathways = pd.read_csv(reactome_path, sep='\t', header=None)
    pathways.columns = ['uniprot_id', 'reactome_id', 'url', 'name', '???', 'organism']
    
    # Filter for human proteins in our protein space
    pathways = pathways[
        (pathways.organism == 'Homo sapiens') & 
        (pathways.uniprot_id.isin(protspace))
    ]
    
    # Create pathway edges
    pathways = pathways.assign(
        source=['PROTEIN__' + x for x in pathways.uniprot_id],
        target=['PATHWAY__' + x for x in pathways.name],
        edge_type='reactome'
    )
    
    path_df = pathways[['source', 'target', 'edge_type']]
    
    # Filter for pathways with minimum size
    pathway_sizes = path_df.groupby('target').size()
    large_pathways = pathway_sizes[pathway_sizes >= min_pathway_size].index
    path_df = path_df[path_df.target.isin(large_pathways)]
    
    print(f"Loaded {len(large_pathways)} pathways with >= {min_pathway_size} proteins")
    return path_df


def create_biological_graph(drugspace, pathspace, graph_depth, include_mirna=True, include_extra=True):
    """Create the full biological interaction graph."""
    print("Building biological interaction network...")
    # Get biological interactions
    bio_names, bio_df = get_bio_interactions(undirected=False, include_mirna=include_mirna, include_extra=include_extra)
    print(f"Translation edges: {bio_df[bio_df.edge_type == 'translation'].shape[0]}")
    return bio_df


def map_uniprot_to_gene_symbols(uniprot_list):
    """Map UniProt IDs to gene symbols."""
    print(f"Mapping {len(uniprot_list)} UniProt IDs to gene symbols...")
    
    uni2symb = {'uniprot': [], 'gene_symbol': []}
    failed = 0
    
    for i, uni in enumerate(uniprot_list):
        print(f'Processing {i+1}/{len(uniprot_list)} | [failed: {failed}/{i}]', end='\r')
        
        try:
            mapped = mapping.map_name(uni, id_type='uniprot', target_id_type='genesymbol')
            for symbol in mapped:
                uni2symb['uniprot'].append(uni)
                uni2symb['gene_symbol'].append(symbol)
        except:
            failed += 1
    
    uni2symb_df = pd.DataFrame(uni2symb).drop_duplicates()
    print(f"Successfully mapped {len(uni2symb_df)} UniProt-symbol pairs ({failed} failed)")
    
    return uni2symb_df


def process_expression_data(expr_path, uni2symb_df, expr_norm, quantile_clip=(0.02,0.98), normalizer_save_path=None, n_top_genes=None, var_threshold=None):
    """Load and process gene expression data."""
    print("Loading gene expression data...")
    
    # Load expression data
    aml_expr = pd.read_csv(expr_path)[['id', 'gene_name', 'fpkm_unstranded']].rename(
        columns={'fpkm_unstranded': 'FPKM'}
    ).assign(id_type='gdc_id')
    
    aml_expr = aml_expr.dropna()
    # Use max to resolve duplicates (usually a zero and a non-zero)
    aml_expr = aml_expr.groupby(['id', 'gene_name', 'id_type']).max().reset_index()
    
    # Filter for genes in our mapping
    aml_expr = aml_expr[aml_expr.gene_name.isin(uni2symb_df.gene_symbol.unique())]
    
    # Convert to wide format first (needed for variance calculation in normalizer)
    aml_expr = aml_expr.pivot(index='id', columns='gene_name', values='FPKM')
    
    # Determine variance selection parameters  
    var_quantile_param = var_threshold
    
    # Apply expression normalization with integrated variance filtering
    normalizer = ExpressionNormalizer(
        expr_norm, 
        quantile_clip=quantile_clip,
        n_top_variable_genes=n_top_genes,
        var_quantile=var_quantile_param
    )
    normalizer.fit(aml_expr)
    aml_expr = normalizer.transform(aml_expr)
    print(f"Applied {expr_norm} normalization")
    
    # Save normalizer if requested
    if normalizer_save_path is not None:
        normalizer.save(normalizer_save_path)
        print(f"Saved expression normalizer to {normalizer_save_path}")
    
    # Map gene symbols to UniProt IDs
    symb2uni = uni2symb_df.set_index('gene_symbol').uniprot.to_dict()
    aml_expr.columns = [f'EXPR__{symb2uni[x]}' for x in aml_expr.columns]
    
    print(f"Processed expression data: {aml_expr.shape[0]} samples x {aml_expr.shape[1]} genes")
    return aml_expr, normalizer


def process_mds_expression_data(mds_expr_path, normalizer, uni2symb_df):
    """Load and process MDS expression data using fitted normalizer."""
    print("Loading MDS expression data...")
    
    # Load MDS expression data - note different column names
    mds_expr = pd.read_csv(mds_expr_path, sep='\t')[['array_id', 'gene_id', 'FPKM']].rename(
        columns={'array_id': 'id', 'gene_id': 'gene_name'}
    ).assign(id_type='mds_id')
    
    # Handle NaN values and drop them
    mds_expr = mds_expr.dropna()
    
    # Use max to resolve duplicates (consistent with AML processing)
    mds_expr = mds_expr.groupby(['id', 'gene_name', 'id_type']).max().reset_index()
    
    # Filter for genes in our UniProt mapping
    mds_expr = mds_expr[mds_expr.gene_name.isin(uni2symb_df.gene_symbol.unique())]
    
    # Convert to wide format
    mds_expr = mds_expr.pivot(index='id', columns='gene_name', values='FPKM')
    
    # Apply the same normalization as AML data using the fitted normalizer
    mds_expr_normalized = normalizer.transform(mds_expr)
    print(f"Applied {normalizer.method} normalization to MDS data")
    
    # Map gene symbols to UniProt IDs (same as AML processing)
    symb2uni = uni2symb_df.set_index('gene_symbol').uniprot.to_dict()
    mds_expr_normalized.columns = [f'EXPR__{symb2uni[x]}' for x in mds_expr_normalized.columns]
    
    print(f"Processed MDS expression data: {mds_expr_normalized.shape[0]} samples x {mds_expr_normalized.shape[1]} genes")
    return mds_expr_normalized


def construct_final_graph(dti_df, bio_df, path_df, aml_expr, drugspace, pathspace, graph_depth):
    """Construct the final heterogeneous graph."""
    print("Constructing final graph...")
    
    # Combine all edge dataframes
    edgedf = pd.concat([dti_df, bio_df, path_df], axis=0)
    
    # Create NetworkX graph
    G = nx.DiGraph()
    for _, row in edgedf.iterrows():
        G.add_edge(row.source, row.target)
    
    print(f"Initial graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Subset graph based on drug roots and pathway leaves
    G = subset_graph(G, depth=graph_depth, roots=drugspace, leafs=pathspace)
    print(f"Subsetted graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Get node sets
    drugspace_final = [d for d in G.nodes() if d.startswith('DRUG__')]
    protspace_final = [p for p in G.nodes() if p.startswith('PROTEIN__')]
    rnaspace_final = [r for r in G.nodes() if r.startswith('RNA__')]
    pathspace_final = [p for p in G.nodes() if p.startswith('PATHWAY__')]
    
    print(f"Final node counts - Drugs: {len(drugspace_final)}, "
          f"Proteins: {len(protspace_final)}, RNAs: {len(rnaspace_final)}, "
          f"Pathways: {len(pathspace_final)}")
    
    # Add expression input nodes and edges
    overlap_unis = set([x.split('__')[1] for x in aml_expr.columns.tolist()]).intersection(
        set([x.split('__')[1] for x in rnaspace_final])
    )
    input_expr = [f'EXPR__{x}' for x in overlap_unis]
    aml_expr = aml_expr[input_expr]
    
    # Add expression edges
    G = G.copy()  # unfreeze graph
    for uni in overlap_unis:
        G.add_edge(f'EXPR__{uni}', f'RNA__{uni}')
    
    # Add output edges (from pathways to output)
    for p in pathspace_final:
        G.add_edge(p, 'OUT_AUC')
    
    # Define node sets for PyG conversion
    input_nodes = aml_expr.columns.tolist() + drugspace_final
    function_nodes = protspace_final + rnaspace_final + pathspace_final
    output_nodes = ['OUT_AUC']
    
    print(f"Node sets - Input: {len(input_nodes)}, "
          f"Function: {len(function_nodes)}, Output: {len(output_nodes)}")
    
    return G, input_nodes, function_nodes, output_nodes, aml_expr, drugspace_final


def filter_drug_data(drug, drugspace, aml_expr, clinical_data_path, train_frac, seed):
    """Filter drug data and assign train/val/test partitions using MDS logic."""
    np.random.seed(seed)
    
    included_drugs = [x.split('__')[1] for x in drugspace]
    included_ids = aml_expr.index.unique().tolist()
    
    # Filter for drugs and samples in our graph
    drug_filtered = drug[
        drug.inhibitor_1.str.lower().isin(included_drugs) & 
        drug.id.isin(included_ids)
    ]
    
    # Include combination agents (nan or included drugs)
    drug_filtered = drug_filtered[
        drug_filtered.inhibitor_2.isna() | 
        drug_filtered.inhibitor_2.str.lower().isin(included_drugs)
    ]
    
    # Assign partitions using MDS logic from notebook
    # Prior MDS ids should be the test set, train/val should be randomly split
    clinical_data = pd.read_csv(clinical_data_path)
    prior_mds_ids = clinical_data[clinical_data.priorMDS == 'y'].gdc_id.unique().tolist()
    
    # First assign train/val randomly using configurable train_frac
    val_frac = 1.0 - train_frac
    partitions = np.random.choice(['train', 'val'], size=len(drug_filtered), p=[train_frac, val_frac])
    drug_filtered = drug_filtered.assign(partition=partitions)
    
    # Then override with test for prior MDS patients
    drug_filtered['partition'] = [
        'test' if sample_id in prior_mds_ids else partition 
        for sample_id, partition in zip(drug_filtered.id, drug_filtered.partition)
    ]
    
    # Count final partitions
    train_count = sum(drug_filtered.partition == 'train')
    val_count = sum(drug_filtered.partition == 'val') 
    test_count = sum(drug_filtered.partition == 'test')
    
    print(f"Final drug data: {len(drug_filtered)} measurements")
    print(f"Train: {train_count} ({train_count/len(drug_filtered)*100:.1f}%), "
          f"Val: {val_count} ({val_count/len(drug_filtered)*100:.1f}%), "
          f"Test: {test_count} ({test_count/len(drug_filtered)*100:.1f}%)")
    print(f"Prior MDS patients assigned to test set: {len(prior_mds_ids)} unique IDs")
    
    return drug_filtered


def print_graph_summary(G, input_nodes, function_nodes, output_nodes, drug_final, aml_expr):
    """Print comprehensive summary of the constructed graph and data."""
    print("\n" + "="*60)
    print("GRAPH CONSTRUCTION SUMMARY")
    print("="*60)
    
    # Graph structure metrics
    print(f"\n📊 GRAPH STRUCTURE:")
    print(f"   Total nodes: {len(G.nodes()):,}")
    print(f"   Total edges: {len(G.edges()):,}")
    print(f"   Average degree: {2*len(G.edges())/len(G.nodes()):.2f}")
    
    # Node type breakdown
    drugspace = [d for d in G.nodes() if d.startswith('DRUG__')]
    protspace = [p for p in G.nodes() if p.startswith('PROTEIN__')]
    rnaspace = [r for r in G.nodes() if r.startswith('RNA__')]
    pathspace = [p for p in G.nodes() if p.startswith('PATHWAY__')]
    exprspace = [e for e in G.nodes() if e.startswith('EXPR__')]
    
    print(f"\n🔗 NODE TYPES:")
    print(f"   Drug nodes: {len(drugspace):,}")
    print(f"   Protein nodes: {len(protspace):,}")
    print(f"   RNA nodes: {len(rnaspace):,}")
    print(f"   Expression input nodes: {len(exprspace):,}")
    print(f"   Pathway nodes: {len(pathspace):,}")
    print(f"   Output nodes: {len(output_nodes):,}")
    
    # Edge type breakdown
    edge_types = {}
    for edge in G.edges(data=True):
        if 'edge_type' in edge[2]:
            edge_type = edge[2]['edge_type']
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        elif edge[0].startswith('EXPR__') and edge[1].startswith('RNA__'):
            edge_types['expression'] = edge_types.get('expression', 0) + 1
        elif edge[1] == 'OUT_AUC':
            edge_types['output'] = edge_types.get('output', 0) + 1
    
    print(f"\n🔀 EDGE TYPES:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"   {edge_type.capitalize()}: {count:,}")
    
    # Data metrics
    print(f"\n📈 DATA METRICS:")
    print(f"   Expression samples: {aml_expr.shape[0]:,}")
    print(f"   Expression features: {aml_expr.shape[1]:,}")
    print(f"   Drug response measurements: {len(drug_final):,}")
    print(f"   Unique drugs: {len(drug_final.inhibitor_1.unique()):,}")
    print(f"   Unique samples: {len(drug_final.id.unique()):,}")
    
    # Split information
    train_count = sum(drug_final.partition == 'train')
    val_count = sum(drug_final.partition == 'val')
    test_count = sum(drug_final.partition == 'test')
    
    print(f"\n🎯 DATA SPLITS:")
    print(f"   Training: {train_count:,} ({train_count/len(drug_final)*100:.1f}%)")
    print(f"   Validation: {val_count:,} ({val_count/len(drug_final)*100:.1f}%)")
    print(f"   Test (MDS): {test_count:,} ({test_count/len(drug_final)*100:.1f}%)")
    
    # Graph connectivity
    if len(G.nodes()) > 0:
        print(f"\n🔀 CONNECTIVITY:")
        print(f"   Graph density: {len(G.edges())/(len(G.nodes())*(len(G.nodes())-1))*100:.4f}%")
        print(f"   Strongly connected: {nx.is_strongly_connected(G)}")
        print(f"   Weakly connected: {nx.is_weakly_connected(G)}")
    
    print("="*60)


def remove_duplicate_edges(G):
    """Remove duplicate edges from a NetworkX graph.
    
    Args:
        G: NetworkX graph object
        
    Returns:
        NetworkX graph with duplicate edges removed
    """
    print("Removing duplicate edges from graph...")
    
    # Get initial edge count
    initial_edges = len(G.edges())
    
    # For directed graphs, create a new graph with unique edges
    G_clean = type(G)()  # Create same type of graph (DiGraph, Graph, etc.)
    
    # Add all nodes first (preserving node attributes if any)
    G_clean.add_nodes_from(G.nodes(data=True))
    
    # Track unique edges
    unique_edges = set()
    edge_data = {}
    
    # Iterate through edges and keep only unique ones
    for source, target, data in G.edges(data=True):
        edge_key = (source, target)
        if edge_key not in unique_edges:
            unique_edges.add(edge_key)
            edge_data[edge_key] = data
    
    # Add unique edges to new graph
    for (source, target), data in edge_data.items():
        G_clean.add_edge(source, target, **data)
    
    final_edges = len(G_clean.edges())
    duplicates_removed = initial_edges - final_edges
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate edges ({initial_edges} -> {final_edges})")
    else:
        print("No duplicate edges found")
    
    return G_clean


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Drug Response Graph Construction ===")
    print(f"Response normalization: {args.resp_norm}")
    print(f"Graph depth: {args.graph_depth}")
    print(f"Output directory: {output_dir}")
    
    # Load and process drug response data
    drug = load_and_process_drug_data(args.drug_data, args.clinical_data, args.resp_norm)
    
    # Get drug candidates for filtering
    drug_candidates = (drug.inhibitor_1.str.lower().unique().tolist() + 
                      drug.inhibitor_2.str.lower().unique().tolist())
    drug_candidates = [x for x in drug_candidates if pd.notna(x)]
    
    # Load drug-target interactions
    dtis = load_drug_target_interactions(
        args.targetome_data, args.drug_meta, drug_candidates, args.max_assay_value
    )
    
    # Create DTI graph edges
    dti_df = create_dti_graph(dtis)
    drugspace = dti_df.source.unique()
    
    # Load biological interactions
    bio_df = create_biological_graph(drugspace, None, args.graph_depth, args.include_mirna, args.include_extra)
    
    # Get protein space from biological interactions
    protspace = [x.split('__')[1] for x in 
                np.unique(bio_df.source.tolist() + bio_df.target.tolist()).tolist() 
                if x.split('__')[0] == 'PROTEIN']
    
    # Load pathway data
    path_df = load_pathway_data(args.reactome_data, protspace, args.min_pathway_size)
    pathspace = path_df.target.unique()
    
    # Get RNA space and map to gene symbols
    rnaspace = [r for r in np.unique(bio_df.source.tolist() + bio_df.target.tolist()) 
               if r.startswith('RNA__')]
    uniprot_rnaspace = [x.split('__')[1] for x in rnaspace]
    uni2symb_df = map_uniprot_to_gene_symbols(uniprot_rnaspace)
    
    # Process expression data
    aml_expr, normalizer = process_expression_data(
        args.expression_data, uni2symb_df, args.expr_norm,
        quantile_clip=tuple(args.expr_clip_quantiles),
        normalizer_save_path=args.expr_normalizer_save_path,
        n_top_genes=args.expr_n_top_genes,
        var_threshold=args.expr_var_threshold
    )
    
    # Process MDS expression data using the same normalizer
    mds_expr_path = '/home/teddy/local/AMLVAE/data/20241219_WTS_Data_Proj805.csv'
    mds_expr = process_mds_expression_data(
        mds_expr_path, normalizer, uni2symb_df
    )
    
    # Construct final graph
    G, input_nodes, function_nodes, output_nodes, aml_expr_final, drugspace_final = construct_final_graph(
        dti_df, bio_df, path_df, aml_expr, drugspace, pathspace, args.graph_depth
    )

    # Remove duplicate edges in graph
    G = remove_duplicate_edges(G)
    
    # Convert to PyG format
    print("Converting to PyTorch Geometric format...")
    data = nx2pyg(G, input_nodes=input_nodes, function_nodes=function_nodes, output_nodes=output_nodes)
    
    # Filter and partition drug data
    drug_final = filter_drug_data(
        drug, drugspace_final, aml_expr_final, 
        args.clinical_data, args.train_frac, args.seed
    )

    # Save outputs
    print("Saving processed data...")
    torch.save(data, output_dir / 'graph.pt')
    aml_expr_final.to_csv(output_dir / 'aml_expr.csv')
    mds_expr.to_csv(output_dir / 'mds_expr.csv')
    drug_final.to_csv(output_dir / 'resp.csv', index=False)
    
    # Print comprehensive summary
    print_graph_summary(G, input_nodes, function_nodes, output_nodes, drug_final, aml_expr_final)
    
    print("\n=== Graph Construction Complete ===")
    print(f"Graph saved to: {output_dir / 'graph.pt'}")
    print(f"AML expression data saved to: {output_dir / 'aml_expr.csv'}")
    print(f"MDS expression data saved to: {output_dir / 'mds_expr.csv'}")
    print(f"Response data saved to: {output_dir / 'resp.csv'}")


if __name__ == '__main__':
    main() 