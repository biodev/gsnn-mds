# Data Processing and Graph Construction

## Overview

The graph construction pipeline transforms raw molecular data into a biologically-informed graph structure suitable for neural network processing. This document outlines the key preprocessing steps and design decisions.

## Input Data

### 1. Expression Data
- **Source**: AML patient gene expression profiles
- **Format**: CSV with patient IDs and gene expression values
- **Processing**: 
  - Log transformation for variance stabilization
  - Missing value handling (zero imputation for unmeasured genes)
  - Standardization across patients

### 2. Drug Response Data
- **Source**: Drug sensitivity screens (IC50 values)
- **Format**: Patient-drug-response triplets
- **Processing**:
  - Log transformation of IC50 values
  - Normalization to zero mean, unit variance (optional: z-score WITHIN each drug)
  - Train/validation/test split by patient-drug combinations

### 3. Biological Networks
- **Source**: OmniPath database
- **Content**: 
  - Protein-protein interactions (PPI)
  - Pathway annotations
  - Drug-target relationships
- **Processing**: 
  - Gene name standardization
  - Network deduplication
  - Confidence score filtering

## Graph Construction Process

### Step 1: Define Node Types

The heterogeneous graph contains three node types:

```
Input Nodes ─────┐
                 ├──→ Function Nodes ──→ Output Node
Drug Nodes ──────┘
```

- **Input Layer**: One node per gene + one node per drug
- **Function Layer**: Intermediate nodes representing biological processes
- **Output Layer**: Single prediction node

### Step 2: Create Function Nodes

Function nodes are created through clustering of the PPI network:
1. Extract PPI subnetwork for expressed genes
2. Apply community detection or pathway-based grouping
3. Create one function node per identified module
4. Prune isolated nodes with no connections

### Step 3: Build Edge Connections

#### Input → Function Edges
- Connect genes to function nodes based on:
  - Direct protein interactions
  - Pathway co-membership
  - Functional annotations

#### Drug → Function Edges
- Connect drugs to function nodes via:
  - Known drug targets
  - Pathway enrichment of targets
  - Pharmacological annotations

#### Function → Function Edges
- Represent inter-pathway communication:
  - Shared proteins between pathways
  - Known pathway crosstalk
  - Regulatory relationships

#### Function → Output Edges
- All function nodes connect to the output node
- Allows integration of all pathway-level information

### Step 4: Feature Initialization

- **Gene Nodes**: Initialized with expression values
- **Drug Nodes**: One-hot encoding (1 for active drug, 0 otherwise)
- **Function Nodes**: Initialized as learnable parameters
- **Output Node**: No initial features (receives messages)

## Data Structure Format

The final graph is stored as a PyTorch Geometric `HeteroData` object:

```python
data = HeteroData()

# Node features
data['input'].x = input_features
data['function'].x = function_features
data['output'].x = output_features

# Edge indices
data['input', 'connects', 'function'].edge_index = input_function_edges
data['drug', 'targets', 'function'].edge_index = drug_function_edges
data['function', 'interacts', 'function'].edge_index = function_function_edges
data['function', 'predicts', 'output'].edge_index = function_output_edges

# Node name mappings
data.node_names_dict = {
    'input': gene_names + drug_names,
    'function': function_names,
    'output': ['prediction']
}
```

## Key Design Decisions

### 1. Sparsity Handling
- Not all genes have expression data → zero imputation
- Not all drugs have complete target profiles → use available annotations
- Missing pathway information → rely on PPI network structure

### 2. Scalability Considerations
- Graph constructed once and reused across samples
- Patient-specific information injected at input nodes only
- Efficient sparse matrix representations

### 3. Biological Interpretability
- Function nodes correspond to real biological processes
- Edge types reflect different biological relationships
- Graph structure preserves pathway hierarchy

## Quality Control Steps

1. **Node Coverage**: Ensure all measured genes are represented
2. **Connectivity**: Verify no isolated subgraphs
3. **Edge Validation**: Cross-reference with multiple databases
4. **Size Constraints**: Balance biological completeness with computational efficiency

## Usage in Training

During training, each sample:
1. Loads the patient's expression profile
2. Sets the appropriate drug indicator to 1
3. Propagates information through the fixed graph structure
4. Produces a drug sensitivity prediction

This design allows the model to learn drug-specific and pathway-specific parameters while maintaining a consistent biological structure across all predictions. 