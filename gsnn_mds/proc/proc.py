import os
import pandas as pd
import numpy as np
import torch
import networkx as nx
from pathlib import Path
from pypath.utils import mapping
import joblib
from scipy import stats
from gsnn.proc.bio import get_bio_interactions
from gsnn.proc.subset import subset_graph
from gsnn_mds.proc.ExpressionNormalizer import ExpressionNormalizer


def load_mut(root): 
    aml_mut1 = pd.read_csv(f'{root}/mut/aml_train.csv')
    aml_mut2 = pd.read_csv(f'{root}/mut/aml_test.csv')
    aml_mut3 = pd.read_csv(f'{root}/mut/aml_validation.csv')
    aml_mut = pd.concat([aml_mut1, aml_mut2, aml_mut3])
    #aml_mut = aml_mut.rename(columns={'dbgap_dnaseq_sample':'sample_id'})
    clin = pd.read_csv(f'{root}/beataml_clinical_for_inputs.csv')[['gdc_id', 'dbgap_dnaseq_sample']].dropna().rename(columns={'gdc_id':'sample_id'}) 
    aml_mut = aml_mut.merge(clin, on='dbgap_dnaseq_sample', how='left')
    aml_mut = aml_mut.assign(source='AML')
    # fill na in case 
    aml_mut = aml_mut.fillna(0)
    aml_features = aml_mut.columns[1:].tolist() 

    mds_mut = pd.read_excel(f'{root}/mut/805_data_20250107.xlsx')
    mds_features = mds_mut.columns[60:].tolist() 
    mds_mut = mds_mut.rename(columns={'MLL ID':'sample_id'})
    
    # fill na and convert "NEGATIVE" to 0 and "POSITIVE" to 1 
    pd.set_option('future.no_silent_downcasting', True)
    mds_mut = mds_mut.fillna(0)
    mds_mut = mds_mut.replace('NEGATIVE', 0)
    mds_mut = mds_mut.replace('POSITIVE', 1) 
    mds_mut = mds_mut.replace('VARIANT', 1)
    mds_mut = mds_mut.replace('QUESTIONABLE', 0)
    mds_mut = mds_mut.replace('INCOMPLETE VARIANT', 0)
    mds_mut = mds_mut.replace('INCOMPLETE POSITIVE', 0)
    mds_mut = mds_mut.replace('INCOMPLETE NEGATIVE', 0)
    mds_mut = mds_mut.replace('N.A.', 0)
    mds_mut = mds_mut.replace('POSITIVE_T2', 0)
    mds_mut = mds_mut.assign(source='MDS')

    mut_features = list(set(aml_features).intersection(set(mds_features)))

    aml_mut = aml_mut[['sample_id', 'source'] + mut_features]
    mds_mut = mds_mut[['sample_id', 'source'] + mut_features] 
    mut = pd.concat([aml_mut, mds_mut], axis=0)

    # convert each mut features to int and check for na 
    for f in mut_features: 
        mut[f] = mut[f].astype(int)
        assert mut[f].isna().sum() == 0, f'{f} has {mut[f].isna().sum()} na'

    return mut, mut_features



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

    uni2gene_df = map_uniprot_to_gene_symbols(dtis.uniprot_id.unique().tolist()) 
    dtis = dtis.merge(uni2gene_df, left_on='uniprot_id', right_on='uniprot', how='inner')
    
    print(f"Loaded {len(dtis)} drug-target interactions for {len(dtis.inhibitor.unique())} drugs")
    return dtis


def create_dti_graph(dtis):
    """Create drug-target interaction edges."""
    dti_df = dtis.assign(
        source=['DRUG__' + x for x in dtis.inhibitor.str.lower()],
        target=['PROTEIN__' + x for x in dtis.gene_symbol],
        edge_type='dti'
    )[['source', 'target', 'edge_type']].drop_duplicates()
    
    return dti_df


def load_pathway_data(reactome_path, protspace, min_pathway_size):
    """Load and filter pathway annotation data.
    
    ReactomePathways.gmt (https://reactome.org/download-data) format: 

    <PathwayName1>   <PathID1>     <Gene2>     ...     <GeneN>
    <PathwayName2>   <PathID2>     <Gene2>     ...     <GeneN>
    ...
    <PathwayNameN>   <PathID3>     <Gene2>     ...     <GeneN>

    e.g., 
    3-Methylcrotonyl-CoA carboxylase deficiency     R-HSA-9909438   MCCC1   MCCC2
    3-hydroxyisobutyryl-CoA hydrolase deficiency    R-HSA-9916722   HIBCH
    3-methylglutaconic aciduria     R-HSA-9914274   AUH
    5-Phosphoribose 1-diphosphate biosynthesis      R-HSA-73843     PRPS1   PRPS1L1 PRPS2

    """
    print("Loading pathway annotations...")

    # Parse GMT format: PathwayName, ReactomeID, Gene1, Gene2, ...
    rows = []
    with open(reactome_path, encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 3:
                continue
            pathway_name = fields[0]
            reactome_id = fields[1]
            genes = [g for g in fields[2:] if g]
            for gene in genes:
                rows.append({'pathway_name': pathway_name, 'reactome_id': reactome_id, 'gene_symbol': gene})

    pathways = pd.DataFrame(rows)

    # Filter for proteins in our protein space
    pathways = pathways[pathways.gene_symbol.isin(protspace)]

    # Create pathway edges (use reactome_id for stable, clean node IDs)
    path_df = pathways.assign(
        source=['PROTEIN__' + x for x in pathways.gene_symbol],
        target=['PATHWAY__' + x for x in pathways.reactome_id],
        edge_type='reactome'
    )[['source', 'target', 'edge_type']].drop_duplicates()

    # Filter for pathways with minimum size
    pathway_sizes = path_df.groupby('target').size()
    large_pathways = pathway_sizes[pathway_sizes >= min_pathway_size].index
    path_df = path_df[path_df.target.isin(large_pathways)]

    print(f"Loaded {len(large_pathways)} pathways with >= {min_pathway_size} proteins")
    return path_df

def map_gene_symbols_to_uniprot(gene_symbols):
    """Map gene symbols to UniProt IDs."""
    print(f"Mapping {len(gene_symbols)} gene symbols to UniProt IDs...")
    
    symb2uni = {'gene_symbol': [], 'uniprot': []}
    failed = 0
    
    for i, symbol in enumerate(gene_symbols):
        print(f'Processing {i+1}/{len(gene_symbols)} | [failed: {failed}/{i}]', end='\r')
        
        try:
            mapped = mapping.map_name(symbol, id_type='genesymbol', target_id_type='uniprot')
            for uni in mapped:
                symb2uni['gene_symbol'].append(symbol)
                symb2uni['uniprot'].append(uni)
        except:
            failed += 1
    
    symb2uni_df = pd.DataFrame(symb2uni).drop_duplicates()
    print(f"Successfully mapped {len(symb2uni_df)} symbol-UniProt pairs ({failed} failed)")
    
    return symb2uni_df


def load_van_galen_edges(data_root, protspace): 
    '''secondary pathway-like edges from van galen (cell type) gene sets'''
    vg_path = os.path.join(data_root, 'van_galen_malig_celltypes_top30.txt')
    vg = pd.read_csv(vg_path, sep='\t')
    vg = vg.rename({'vg_type':'dst', 'display_label':'src'}, axis=1)[['src', 'dst']]
    
    # Filter for proteins in our protein space (protspace contains UniProt IDs)
    vg = vg[vg.src.isin(protspace)]
    
    # Format as pathway edges similar to reactome data
    vg = vg.assign(
        source=['PROTEIN__' + x for x in vg.src],
        target=['PATHWAY__VG_' + x for x in vg.dst],  # VG_ prefix to distinguish van galen pathways
        edge_type='van_galen'
    )
    
    vg_df = vg[['source', 'target', 'edge_type']]
    print(f"Loaded {len(vg_df)} van galen edges connecting to {len(vg_df.target.unique())} van galen pathways")
    return vg_df


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


def process_expression_data(expr_path, gene_symbols, expr_norm, 
                            quantile_clip=(0.02,0.98), normalizer_save_path=None, 
                            n_top_genes=None, var_threshold=None):

    """Load and process gene expression data."""
    print("Loading gene expression data...")
    
    # Load expression data
    aml_expr = pd.read_csv(expr_path)[['id', 'gene_name', 'fpkm_unstranded']].rename(
        columns={'fpkm_unstranded': 'FPKM'}
    ).assign(id_type='gdc_id')
    
    aml_expr = aml_expr.dropna()
    # Use max to resolve duplicates (usually a zero and a non-zero)
    aml_expr = aml_expr.groupby(['id', 'gene_name', 'id_type']).max().reset_index()

    if gene_symbols is None: 
        print("No gene symbols provided - using all genes from expression data")
        gene_symbols = aml_expr.gene_name.unique().tolist()
    
    # Filter for genes in our mapping
    aml_expr = aml_expr[
        aml_expr.gene_name.isin(gene_symbols)
    ]
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
    aml_expr.columns = [f'EXPR__{x}' for x in aml_expr.columns]
    
    print(f"Processed expression data: {aml_expr.shape[0]} samples x {aml_expr.shape[1]} genes")
    return aml_expr, normalizer


def process_mds_expression_data(mds_expr_path, normalizer):
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
    
    # NOTE: this is taken care of in the Normalizer.transform() method
    # Filter for genes in our UniProt mapping
    #mds_expr = mds_expr[mds_expr.gene_name.isin(gene_symbols)]
    
    # Convert to wide format
    mds_expr = mds_expr.pivot(index='id', columns='gene_name', values='FPKM')
    
    # Apply the same normalization as AML data using the fitted normalizer
    mds_expr_normalized = normalizer.transform(mds_expr)
    print(f"Applied {normalizer.method} normalization to MDS data")
    
    # Map gene symbols to UniProt IDs (same as AML processing)
    mds_expr_normalized.columns = [f'EXPR__{x}' for x in mds_expr_normalized.columns]
    
    print(f"Processed MDS expression data: {mds_expr_normalized.shape[0]} samples x {mds_expr_normalized.shape[1]} genes")
    return mds_expr_normalized


def process_mutation_data(mut, mut_features, uni2symb_df, sample_ids_overlap):
    """Process mutation data for AML samples and map to UniProt IDs."""
    print("Processing AML mutation data...")
    
    # Filter mutation data to only include overlapping sample IDs
    mut_filtered = mut[mut['sample_id'].isin(sample_ids_overlap)]
    
    # Select only mutation features + sample_id
    mut_filtered = mut_filtered[['sample_id'] + mut_features]
    
    # Set sample_id as index for alignment with expression data
    mut_filtered = mut_filtered.set_index('sample_id')
    
    # Map gene symbols to UniProt IDs
    # First create a mapping from mut_features (gene symbols) to UniProt IDs
    symb2uni_mut = {}
    for feature in mut_features:
        # Try to find UniProt mapping for this gene symbol
        matching_rows = uni2symb_df[uni2symb_df['gene_symbol'] == feature]
        if len(matching_rows) > 0:
            # Take first UniProt ID if multiple exist
            uniprot_id = matching_rows.iloc[0]['uniprot']
            symb2uni_mut[feature] = uniprot_id
    
    # Filter mutation features to only those with UniProt mappings
    mapped_features = [f for f in mut_features if f in symb2uni_mut]
    mut_mapped = mut_filtered[mapped_features]
    
    # Rename columns to MUT__ prefix with UniProt IDs
    mut_mapped.columns = [f'MUT__{symb2uni_mut[x]}' for x in mut_mapped.columns]
    
    print(f"Processed AML mutation data: {mut_mapped.shape[0]} samples x {mut_mapped.shape[1]} mutation features")
    print(f"Successfully mapped {len(mapped_features)} out of {len(mut_features)} mutation features to UniProt IDs")
    
    return mut_mapped


def process_all_mutation_data(mut, mut_features, uni2symb_df):
    """Process mutation data for ALL patients (AML and MDS) and map to UniProt IDs."""
    print("Processing mutation data for all patients (AML + MDS)...")
    
    # Select only mutation features + sample_id + source
    mut_filtered = mut[['sample_id', 'source'] + mut_features].copy()
    
    # Set sample_id as index
    mut_filtered = mut_filtered.set_index('sample_id')
    
    # Map gene symbols to UniProt IDs
    symb2uni_mut = {}
    for feature in mut_features:
        # Try to find UniProt mapping for this gene symbol
        matching_rows = uni2symb_df[uni2symb_df['gene_symbol'] == feature]
        if len(matching_rows) > 0:
            # Take first UniProt ID if multiple exist
            uniprot_id = matching_rows.iloc[0]['uniprot']
            symb2uni_mut[feature] = uniprot_id
    
    # Filter mutation features to only those with UniProt mappings
    mapped_features = [f for f in mut_features if f in symb2uni_mut]
    mut_mapped = mut_filtered[['source'] + mapped_features].copy()
    
    # Rename mutation columns to MUT__ prefix with UniProt IDs
    rename_dict = {f: f'MUT__{symb2uni_mut[f]}' for f in mapped_features}
    mut_mapped = mut_mapped.rename(columns=rename_dict)
    
    # Separate AML and MDS mutation data
    aml_mut = mut_mapped[mut_mapped['source'] == 'AML'].drop('source', axis=1)
    mds_mut = mut_mapped[mut_mapped['source'] == 'MDS'].drop('source', axis=1)
    
    print(f"Processed mutation data:")
    print(f"  - AML: {aml_mut.shape[0]} samples x {aml_mut.shape[1]} mutation features")
    print(f"  - MDS: {mds_mut.shape[0]} samples x {mds_mut.shape[1]} mutation features")
    print(f"  - Successfully mapped {len(mapped_features)} out of {len(mut_features)} mutation features")
    
    return aml_mut, mds_mut, list(rename_dict.values())


def construct_final_graph(dti_df, bio_df, path_df, omics_df, drugspace, pathspace, graph_depth, vg_df=None):
    """Construct the final heterogeneous graph."""
    print("Constructing final graph...")
    
    # Combine all edge dataframes
    edge_dfs = [dti_df, bio_df, path_df]
    if vg_df is not None:
        edge_dfs.append(vg_df)
        print(f"Including {len(vg_df)} van galen edges")
    
    edgedf = pd.concat(edge_dfs, axis=0)
    
    # Create NetworkX graph
    G = nx.DiGraph()
    for _, row in edgedf.iterrows():
        G.add_edge(row.source, row.target)
    
    print(f"Initial graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Subset graph based on drug roots and pathway leaves
    G = subset_graph(G, depth=graph_depth, roots=drugspace, leafs=pathspace)
    G = G.copy() # unfreeze 
    print(f"Subsetted graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Get node sets
    drugspace_final = [d for d in G.nodes() if d.startswith('DRUG__')]
    protspace_final = [p for p in G.nodes() if p.startswith('PROTEIN__')]
    rnaspace_final = [r for r in G.nodes() if r.startswith('RNA__')]
    pathspace_final = [p for p in G.nodes() if p.startswith('PATHWAY__')]
    
    print(f"Final node counts - Drugs: {len(drugspace_final)}, "
          f"Proteins: {len(protspace_final)}, RNAs: {len(rnaspace_final)}, "
          f"Pathways: {len(pathspace_final)}")

    rnaspace_uniprots = [x.split('__')[1] for x in rnaspace_final]
    protspace_uniprots = [x.split('__')[1] for x in protspace_final]
    
    # Add expr/mut edges 
    omic_nodes = [] 
    for g in omics_df.columns:
        node_type, uniprot = g.split('__') 
        zz = 0 
        if uniprot in rnaspace_uniprots:
            G.add_edge(f'{node_type}__{uniprot}', f'RNA__{uniprot}')
            zz += 1 
        if uniprot in protspace_uniprots:
            G.add_edge(f'{node_type}__{uniprot}', f'PROTEIN__{uniprot}')
            zz += 1 
        if zz > 0:
            omic_nodes.append(g)
    
    # Add output edges (from pathways to output)
    for p in pathspace_final:
        G.add_edge(p, 'OUT_AUC')
    
    # Define node sets for PyG conversion
    input_nodes = omic_nodes + drugspace_final
    
    function_nodes = protspace_final + rnaspace_final + pathspace_final
    output_nodes = ['OUT_AUC']
    
    print(f"Node sets - Input: {len(input_nodes)}, "
          f"Function: {len(function_nodes)}, Output: {len(output_nodes)}")

    drug_df_ = pd.DataFrame({d:np.zeros(len(omics_df)) for d in drugspace_final})
    inputs_df = pd.concat([omics_df, drug_df_], axis=1)
    inputs_df = inputs_df[input_nodes].fillna(0) 

    # Return mutation data along with other outputs
    return G, input_nodes, function_nodes, output_nodes, drugspace_final, inputs_df


def filter_drug_data(drug, drugspace, ids, clinical_data_path, train_frac, seed):
    """Filter drug data and assign train/val/test partitions using MDS logic."""
    np.random.seed(seed)
    
    included_drugs = [x.split('__')[1] for x in drugspace]
    
    # Filter for drugs and samples in our graph
    drug_filtered = drug[
        drug.inhibitor_1.str.lower().isin(included_drugs) & 
        drug.id.isin(ids)
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


def create_patient_data_dict(data, inputs_df):
    """
    Create a dictionary mapping patient_id -> tensor aligned with input_nodes.
    
    Args:
        data: Graph data object with node_names_dict
        inputs_df: Merged dataframe with EXPR__ and MUT__ columns matching graph input nodes
        
    Returns:
        dict: patient_id -> torch.tensor aligned with input nodes
    """

    patient_data = {}
    
    for patient_id in inputs_df.index:
        x = torch.zeros(len(data.node_names_dict['input']), dtype=torch.float32)
        
        for i, node in enumerate(data.node_names_dict['input']):
            # Only process patient input nodes (EXPR__ and MUT__)
            if node.startswith(('EXPR__', 'MUT__')):
                if node in inputs_df.columns:
                    x[i] = float(inputs_df.loc[patient_id, node])
                # If node not in inputs_df, leave as 0 (for missing mutation features)
            # Drug nodes and other nodes remain 0 (will be set by AMLDataset)
            
        patient_data[patient_id] = x 

    # Count feature types for summary
    expr_features = len([col for col in inputs_df.columns if col.startswith('EXPR__')])
    mut_features = len([col for col in inputs_df.columns if col.startswith('MUT__')])
    
    print(f"Created patient data dictionary with {len(patient_data)} patients")
    print(f"  - Expression features: {expr_features}")
    print(f"  - Mutation features: {mut_features}")
    print(f"  - Total input features: {expr_features + mut_features}")

    return patient_data 


def create_drug_response_dataframe(drug_final):
    """
    Create a clean dataframe linking patient_id to drug response data.
    
    Args:
        drug_final: Filtered drug response dataframe
        
    Returns:
        pd.DataFrame: Columns include patient_id, drug info, response (AUC), and partition
    """
    # Select relevant columns and rename for clarity
    response_df = drug_final[[
        'id', 'inhibitor_1', 'inhibitor_2', 'auc', 'partition'
    ]].copy()
    
    response_df = response_df.rename(columns={
        'id': 'patient_id',
        'inhibitor_1': 'drug_1', 
        'inhibitor_2': 'drug_2',
        'auc': 'response'
    })
    
    print(f"Created drug response dataframe with {len(response_df)} measurements")
    print(f"  - Train: {sum(response_df.partition == 'train')}")
    print(f"  - Val: {sum(response_df.partition == 'val')}")
    print(f"  - Test: {sum(response_df.partition == 'test')}")
    print(f"  - Unique patients: {len(response_df.patient_id.unique())}")
    
    return response_df


def load_processed_data(data_dir):
    """
    Load the structured data for training.
    
    Args:
        data_dir: Directory containing the processed data files
        
    Returns:
        tuple: (patient_data_dict, drug_response_df, graph_data, patient_input_nodes)
    """
    data_dir = Path(data_dir)
    
    # Load patient data
    patient_data = joblib.load(data_dir / 'patient_data.pkl')
    
    # Load patient input nodes for tensor alignment
    patient_input_nodes = joblib.load(data_dir / 'patient_input_nodes.pkl')
    
    # Load drug responses  
    drug_response_df = pd.read_csv(data_dir / 'drug_responses.csv')
    
    # Load graph
    graph_data = torch.load(data_dir / 'graph.pt')
    
    print(f"Loaded processed data from {data_dir}")
    print(f"  - {len(patient_data)} patients")
    print(f"  - {len(drug_response_df)} drug response measurements")
    print(f"  - Patient tensor size: {next(iter(patient_data.values())).shape[0]}")
    print(f"  - Patient input nodes: {len(patient_input_nodes)}")
    
    return patient_data, drug_response_df, graph_data, patient_input_nodes


def get_training_tensors(patient_data, drug_response_df, partition='train'):
    """
    Get training data tensors for a specific partition.
    
    Args:
        patient_data: Patient data dictionary mapping patient_id -> tensor
        drug_response_df: Drug response dataframe
        partition: 'train', 'val', or 'test'
        
    Returns:
        tuple: (X_tensor, y_tensor, sample_info) where:
            - X_tensor: torch.Tensor of shape (n_samples, n_features)
            - y_tensor: torch.Tensor of responses
            - sample_info: dataframe with drug and patient info for each sample
    """
    # Filter for the specified partition
    partition_data = drug_response_df[drug_response_df.partition == partition].copy()
    
    # Extract patient tensors and responses for each sample
    patient_tensors = []
    responses = []
    valid_samples = []
    
    for idx, row in partition_data.iterrows():
        patient_id = row['patient_id']
        if patient_id in patient_data:
            patient_tensors.append(patient_data[patient_id])
            responses.append(row['response'])
            valid_samples.append(idx)
    
    if not patient_tensors:
        raise ValueError(f"No valid samples found for partition '{partition}'")
    
    # Stack tensors and create response tensor
    X_tensor = torch.stack(patient_tensors)
    y_tensor = torch.tensor(responses, dtype=torch.float32)
    sample_info = partition_data.loc[valid_samples]
    
    print(f"Retrieved {partition} data:")
    print(f"  - {len(X_tensor)} samples")
    print(f"  - {X_tensor.shape[1]} features per sample")
    print(f"  - {len(sample_info.patient_id.unique())} unique patients")
    
    return X_tensor, y_tensor, sample_info


def save_processed_data(output_dir, patient_data, drug_response_df, graph_data, patient_input_nodes):
    """
    Save all processed data in the new structured format.
    
    Args:
        output_dir: Output directory path
        patient_data: Dictionary mapping patient_id -> tensor
        drug_response_df: Dataframe linking patient_id to drug response
        graph_data: PyTorch Geometric graph data
        patient_input_nodes: List of patient input node names (for tensor alignment)
    """
    print("Saving processed data in new structured format...")
    
    # Save patient input data as pickle (preserves tensor structure)
    patient_data_path = output_dir / 'patient_data.pkl'
    joblib.dump(patient_data, patient_data_path)
    
    # Save patient input node order for tensor alignment
    patient_nodes_path = output_dir / 'patient_input_nodes.pkl'
    joblib.dump(patient_input_nodes, patient_nodes_path)
    
    # Save drug response dataframe as CSV
    drug_response_path = output_dir / 'drug_responses.csv'
    drug_response_df.to_csv(drug_response_path, index=False)
    
    # Save graph (unchanged)
    graph_path = output_dir / 'graph.pt'
    torch.save(graph_data, graph_path)
    
    print(f"✓ Patient data saved to: {patient_data_path}")
    print(f"✓ Patient input nodes saved to: {patient_nodes_path}")
    print(f"✓ Drug responses saved to: {drug_response_path}")
    print(f"✓ Graph saved to: {graph_path}")
    
    # Print data summary
    print(f"\nData Summary:")
    print(f"  - Total patients: {len(patient_data)}")
    print(f"  - Drug response measurements: {len(drug_response_df)}")
    print(f"  - Patients with drug responses: {len(drug_response_df.patient_id.unique())}")
    
    # Check tensor shape and feature breakdown
    if patient_data:
        sample_tensor = next(iter(patient_data.values()))
        tensor_size = sample_tensor.shape[0]
        
        # Count expression vs mutation features
        expr_features = [node for node in patient_input_nodes if node.startswith('EXPR__')]
        mut_features = [node for node in patient_input_nodes if node.startswith('MUT__')]
        
        print(f"  - Patient tensor size: {tensor_size}")
        print(f"  - Expression features: {len(expr_features)}")
        print(f"  - Mutation features: {len(mut_features)}")
        print(f"  - Total patient input nodes: {len(patient_input_nodes)}")


def print_graph_summary(G, input_nodes, function_nodes, output_nodes, drug_final, aml_expr, aml_mut=None):
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
    mutspace = [m for m in G.nodes() if m.startswith('MUT__')]
    
    print(f"\n🔗 NODE TYPES:")
    print(f"   Drug nodes: {len(drugspace):,}")
    print(f"   Protein nodes: {len(protspace):,}")
    print(f"   RNA nodes: {len(rnaspace):,}")
    print(f"   Expression input nodes: {len(exprspace):,}")
    if len(mutspace) > 0:
        print(f"   Mutation input nodes: {len(mutspace):,}")
    print(f"   Pathway nodes: {len(pathspace):,}")
    print(f"   Output nodes: {len(output_nodes):,}")
    
    # Edge type breakdown
    edge_types = {}
    for edge in G.edges(data=True):
        if 'edge_type' in edge[2]:
            edge_type = edge[2]['edge_type']
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        elif edge[0].startswith('EXPR__') and (edge[1].startswith('RNA__') or edge[1].startswith('PROTEIN__')):
            edge_types['expression'] = edge_types.get('expression', 0) + 1
        elif edge[0].startswith('MUT__') and (edge[1].startswith('RNA__') or edge[1].startswith('PROTEIN__')):
            edge_types['mutation'] = edge_types.get('mutation', 0) + 1
        elif edge[1] == 'OUT_AUC':
            edge_types['output'] = edge_types.get('output', 0) + 1
    
    print(f"\n🔀 EDGE TYPES:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"   {edge_type.capitalize()}: {count:,}")
    
    # Data metrics
    print(f"\n📈 DATA METRICS:")
    print(f"   Expression samples: {aml_expr.shape[0]:,}")
    print(f"   Expression features: {aml_expr.shape[1]:,}")
    if aml_mut is not None:
        print(f"   Mutation samples: {aml_mut.shape[0]:,}")
        print(f"   Mutation features: {aml_mut.shape[1]:,}")
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