# GSNN-MDS: Graph Structured Neural Networks for Multi-Drug Sensitivity Prediction

## Overview

This project predicts drug sensitivity in Acute Myeloid Leukemia (AML) patients by integrating gene expression data with biological pathway knowledge using Graph Structured Neural Networks (GSNN).

## Core Approach

### 1. Data Integration Strategy

The method combines multiple data sources into a unified graph representation:
- **Patient Data**: Gene expression profiles from AML patients
- **Drug Information**: Chemical compounds tested for sensitivity
- **Biological Networks**: Protein-protein interactions and pathway relationships from OmniPath

### 2. Graph Construction Pipeline

#### Node Types in the Heterogeneous Graph:
1. **Input Nodes**
   - Gene expression nodes (one per measured gene)
   - Drug indicator nodes (one per tested compound)

2. **Function Nodes**
   - Intermediate computational nodes representing biological processes
   - Created based on pathway membership and interaction patterns

3. **Output Node**
   - Single prediction node for drug sensitivity score

#### Edge Construction:
- **Expression → Function**: Connects genes to their associated biological functions
- **Drug → Function**: Links drugs to their known targets and affected pathways
- **Function → Function**: Represents pathway crosstalk and protein interactions
- **Function → Output**: Aggregates information for final prediction

### 3. Key Preprocessing Steps

1. **Gene Expression Processing**
   - Missing value imputation (zeros for unmeasured genes)
   - Feature normalization to ensure consistent scale
   - Mapping to standardized gene identifiers

2. **Network Construction**
   - Protein-protein interactions from OmniPath database
   - Pathway membership information integration
   - Drug-target relationship mapping
   - Automatic graph pruning to remove isolated nodes

3. **Data Structuring**
   - Conversion to PyTorch Geometric format
   - Separate edge indices for different edge types
   - Node feature initialization based on data availability

### 4. Training Data Organization

- **Input Format**: Each training sample combines a patient's expression profile with a single drug indicator
- **Output**: Continuous drug sensitivity score (normalized IC50 values)
- **Data Split**: Stratified by patient/drug combinations to ensure robust evaluation

## Biological Rationale

The graph structure captures the hierarchical organization of cellular processes:
- Gene expression changes propagate through protein interactions
- Drug effects are mediated through specific molecular targets
- Pathway-level integration provides robustness to measurement noise
- The architecture naturally handles missing data and incomplete annotations

## Advantages of This Approach

1. **Interpretability**: Graph structure reflects known biology
2. **Flexibility**: Easy incorporation of new data types or relationships
3. **Generalization**: Leverages pathway knowledge to predict for new drugs/patients
4. **Sparsity Handling**: Gracefully manages incomplete molecular profiles

## Implementation Highlights

- Built on PyTorch Geometric for efficient graph neural network operations
- Modular design allows easy swapping of graph construction strategies
- Scalable to larger patient cohorts and drug libraries
- Maintains biological interpretability while achieving predictive performance 