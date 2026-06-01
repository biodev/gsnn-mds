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


--- resulting graph structure --- 

input nodes: drugs, muts, expr 
function nodes: protein-protein interactions, pathway membership, cell type pathways 
output nodes: drug response (1)

--- 

only AML patients have drug response data 

MDS patients will be used for inference.

"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import networkx as nx
from pathlib import Path

from gsnn.simulate.nx2pyg import nx2pyg
from gsnn.proc.bio import get_bio_interactions
from gsnn_mds.proc.proc import *




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create biological graph for drug response prediction')
    
    # Data paths
    parser.add_argument('--data-root', type=str,
                       default='/home/teddy/local/data',
                       help='Root directory containing all data files')
    
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
    parser.add_argument('--train-frac', type=float, default=0.9,
                       help='Fraction of non-MDS samples for training (remainder goes to validation)')
    
    # Note: Test set assignment uses BeatAML-specific logic:
    # - Prior MDS patients are automatically assigned to test set
    # - Remaining samples split between train/val based on --train-frac
    
    # Output paths
    parser.add_argument('--output-dir', type=str, default='../proc',
                       help='Output directory for processed data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    parser.add_argument('--include-mut', type=lambda x: (str(x).lower() in ['true','1','yes','t']), default=True,
                      help='Include mutation features (default: True)')
    parser.add_argument('--include-expr', type=lambda x: (str(x).lower() in ['true','1','yes','t']), default=True,
                      help='Include expression data (default: True)')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print('----------------------------------------------------')
    print(args)
    print('----------------------------------------------------')
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process drug response data
    drug_data_path = os.path.join(args.data_root, 'beataml_probit_curve_fits_v4_distr.txt')
    clinical_data_path = os.path.join(args.data_root, 'beataml_clinical_for_inputs.csv')
    drug = load_and_process_drug_data(drug_data_path, clinical_data_path, args.resp_norm)
    
    # Get drug candidates for filtering
    drug_candidates = (drug.inhibitor_1.str.lower().unique().tolist() + 
                      drug.inhibitor_2.str.lower().unique().tolist())
    drug_candidates = [x for x in drug_candidates if pd.notna(x)]
    
    # Load drug-target interactions
    targetome_data_path = os.path.join(args.data_root, 'targetome_extended-01-23-25.csv')
    drug_meta_path = os.path.join(args.data_root, 'beataml_drugs_for_targetome.csv')
    dtis = load_drug_target_interactions(
        targetome_data_path, drug_meta_path, drug_candidates, args.max_assay_value
    )
    
    # Create DTI graph edges
    dti_df = create_dti_graph(dtis)
    drugspace = dti_df.source.unique()
    
    # Load biological interactions 
    bio_names, bio_df = get_bio_interactions(undirected=False, 
                                     include_mirna=args.include_mirna, 
                                     include_extra=args.include_extra, 
                                     dorothea_levels=['A', 'B'], 
                                     gene_symbol=True, 
                                     verbose=True)

    bio_df = bio_df.rename(columns={'src':'source', 'dst':'target'})
    
    # Get protein space from biological interactions
    protspace = [x.split('__')[1] for x in 
                np.unique(bio_df.source.tolist() + bio_df.target.tolist()).tolist() 
                if x.split('__')[0] == 'PROTEIN']

    rnaspace = [x.split('__')[1] for x in 
                np.unique(bio_df.source.tolist() + bio_df.target.tolist()).tolist() 
                if x.split('__')[0] == 'RNA']
    
    # Load pathway data
    reactome_data_path = os.path.join(args.data_root, 'ReactomePathways.gmt')
    path_df = load_pathway_data(reactome_data_path, protspace, args.min_pathway_size)
    
    # Load van galen edges (cell type pathways)
    vg_df = load_van_galen_edges(args.data_root, protspace)
    
    # Combine pathspaces (reactome + van galen)
    pathspace = np.concatenate([path_df.target.unique(), vg_df.target.unique()])
    
    if args.include_expr:
        print("Processing expression data...")
        # Process expression data
        expression_data_path = os.path.join(args.data_root, 'aml_full_manuscript.csv')
        aml_expr, normalizer = process_expression_data(
            expression_data_path, rnaspace, args.expr_norm,
            quantile_clip=tuple(args.expr_clip_quantiles),
            normalizer_save_path=args.expr_normalizer_save_path,
            n_top_genes=args.expr_n_top_genes,
            var_threshold=args.expr_var_threshold
        )
        
        # Process MDS expression data using the same normalizer
        mds_expr_path = os.path.join(args.data_root, '20241219_WTS_Data_Proj805.csv')
        mds_expr = process_mds_expression_data(
            mds_expr_path, normalizer
        )

        expr = pd.concat([aml_expr, mds_expr])
    else:
        print("NOT including expression data...")
        expr = None
    
    if args.include_mut:
        ## ----------------------------------------------------------------------------
        print("Processing mutation features...")
        # Load mutation data (contains both AML and MDS)
        mut, mut_features = load_mut(args.data_root)
        mut2 = mut.drop(columns=['source']).set_index('sample_id')
        #uni2symb_dict = uni2symb_df.set_index('gene_symbol').to_dict()['uniprot']
        keep_cols = [x for x in mut2.columns if x in protspace]
        mut2 = mut2[keep_cols]
        mut2.columns = ['MUT__' + x for x in mut2.columns]
    ## ---------------------------------------------------------------------------- 
    else: 
        print("NOT including mutation features...")
        mut = None

    if (expr is not None) and (mut is not None):
        omics_df = expr.merge(mut2, left_index=True, right_index=True)
    elif expr is not None:
        omics_df = expr
    elif mut is not None:
        omics_df = mut2
    else:
        raise ValueError("Neither expression nor mutation data provided - MUST PROVIDE ONE OF THE TWO")

    
    # Construct final graph
    G, input_nodes, function_nodes, output_nodes, drugspace_final, inputs_df = construct_final_graph(
        dti_df, bio_df, path_df, omics_df, drugspace, pathspace, args.graph_depth, vg_df=vg_df)

    # Remove duplicate edges in graph
    G = remove_duplicate_edges(G)
    
    # Convert to PyG format
    print("Converting to PyTorch Geometric format...")
    data = nx2pyg(G, input_nodes=input_nodes, function_nodes=function_nodes, output_nodes=output_nodes)
    
    # Filter and partition drug data
    drug_final = filter_drug_data(
        drug, drugspace_final, inputs_df.index.tolist(),
        clinical_data_path, args.train_frac, args.seed
    )

    # Save all outputs
    print("Saving outputs...")
    
    # Save graph data
    torch.save(data, output_dir / 'graph.pt')
    print(f"Saved graph to: {output_dir / 'graph.pt'}")
    
    # Save drug response data
    drug_final.to_csv(output_dir / 'resp.csv', index=False)
    print(f"Saved drug response data to: {output_dir / 'resp.csv'}")
    
    # Save inputs dataframe (contains both expression and mutation features)
    inputs_df.to_csv(output_dir / 'inputs.csv')
    print(f"Saved inputs dataframe to: {output_dir / 'inputs.csv'}")
    
    # Save just AML expression data for backwards compatibility
    aml_expr.to_csv(output_dir / 'aml_expr.csv')
    print(f"Saved AML expression data to: {output_dir / 'aml_expr.csv'}")
    
    # Save MDS expression data separately  
    mds_expr.to_csv(output_dir / 'mds_expr.csv')
    print(f"Saved MDS expression data to: {output_dir / 'mds_expr.csv'}")
    
    print("Data processing and saving completed successfully!")



if __name__ == '__main__':
    main() 