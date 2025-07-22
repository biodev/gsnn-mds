# GSNN-MDS Documentation

This folder contains documentation for the Graph Structured Neural Networks for Multi-Drug Sensitivity (GSNN-MDS) project.

## Documentation Structure

### 1. [Approach Overview](./approach_overview.md)
High-level overview of the project methodology, suitable for quickly understanding the core approach and biological rationale.

### 2. [Data Processing](./data_processing.md)
Detailed documentation of the graph construction pipeline, including:
- Input data formats and preprocessing steps
- Graph node and edge type definitions
- Biological network integration
- Data structure specifications

## Quick Start

For a rapid understanding of the project:
1. Read the **Approach Overview** for the conceptual framework
2. Consult **Data Processing** for implementation details of the graph construction

## Key Concepts

- **GSNN**: Graph Structured Neural Networks that encode biological relationships
- **Multi-Drug Sensitivity**: Predicting patient-specific responses to multiple drugs
- **AML**: Acute Myeloid Leukemia, the cancer type studied
- **OmniPath**: Database providing protein interactions and pathway information

## Project Structure

```
gsnn-mds/
├── notebooks/
│   ├── make_graph.ipynb    # Graph construction pipeline
│   └── train.ipynb         # Model training and evaluation
├── proc/                   # Processed data files
│   ├── aml_expr.csv       # Expression data
│   ├── resp.csv           # Drug response data
│   └── graph.pt           # Constructed graph structure
└── docs/                   # This documentation
```

## Contact

For questions about the methodology or implementation, please refer to the detailed documentation or examine the source notebooks. 