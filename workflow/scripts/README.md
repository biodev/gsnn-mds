# GSNN-MDS Scripts

This directory contains scripts for graph construction and model training for the GSNN-MDS project.

## Overview

The workflow consists of three main steps:
1. **Graph Construction** (`make_graph.py`) - Build biological graph from raw data
2. **GSNN Training** (`train_gsnn.py`) - Train graph structured neural network
3. **NN Training** (`train_nn.py`) - Train baseline neural network for comparison

## make_graph.py

This script converts the notebook functionality for creating biological graphs for drug response prediction into a clean, configurable command-line tool.

### Usage

Basic usage with default parameters:
```bash
python make_graph.py
```

With custom parameters:
```bash
python make_graph.py \
    --resp-norm zscore \
    --expr-norm zscore \
    --graph-depth 5 \
    --expr-var-quantile 0.1 \
    --train-frac 0.8 \
    --output-dir ../proc \
    --seed 42
```

### Key Parameters

**Data Processing:**
- `--resp-norm`: Response normalization method (zscore, unit_norm, within_drug_zscore, none)
- `--expr-norm`: Expression normalization method (unit_norm: 0-1 scaling, zscore: z-score normalization)
- `--expr-var-quantile`: Minimum variance quantile for gene expression filtering (default: 0.1)

**Graph Construction:**
- `--graph-depth`: Maximum depth for graph subsetting (default: 5)
- `--min-pathway-size`: Minimum number of proteins per pathway (default: 25)
- `--max-assay-value`: Maximum Ki/Kd value for drug-target interactions (default: 1000)

**Data Splits:**
- `--train-frac`: Fraction of non-MDS samples for training (default: 0.85, remainder goes to validation)
- Note: Test set uses BeatAML-specific logic - prior MDS patients are automatically assigned to test set

### Outputs

The script generates three main files in the output directory:
- `graph.pt`: PyTorch Geometric graph object
- `aml_expr.csv`: Processed gene expression data 
- `resp.csv`: Drug response data with train/val/test partitions

The script also prints a comprehensive summary at the end showing:
- Graph structure metrics (nodes, edges, connectivity)
- Node type breakdown (drugs, proteins, RNAs, pathways, etc.)
- Edge type breakdown (DTI, protein-protein, transcriptional, etc.)
- Data metrics (samples, features, measurements)
- Train/validation/test split statistics

## train_gsnn.py

This script trains a Graph Structured Neural Network (GSNN) on the biological graph constructed by `make_graph.py`.

### Usage

Basic usage with default parameters:
```bash
python train_gsnn.py
```

With custom hyperparameters:
```bash
python train_gsnn.py \
    --channels 8 \
    --layers 6 \
    --lr 5e-4 \
    --dropout 0.1 \
    --batch-size 128 \
    --epochs 50 \
    --output-dir ../results/gsnn_experiment
```

### Key Parameters

**Model Architecture:**
- `--channels`: Number of channels in GSNN layers (default: 5)
- `--layers`: Number of GSNN layers (default: 8)
- `--dropout`: Dropout probability (default: 0.05)
- `--nonlin`: Activation function (ELU, ReLU, LeakyReLU, GELU, default: ELU)
- `--norm`: Normalization method (batch, layer, none, default: batch)
- `--residual`: Use residual connections (default: True)
- `--node-attn`: Use node attention mechanism (default: True)

**Training Configuration:**
- `--lr`: Learning rate (default: 1e-3)
- `--weight-decay`: Weight decay for regularization (default: 1e-2)
- `--batch-size`: Batch size (default: 256)
- `--epochs`: Maximum epochs (default: 100)
- `--patience`: Early stopping patience (default: 10)

### Outputs

The script generates the following files in the output directory:
- `gsnn_model.pt`: Trained GSNN model
- `gsnn_predictions.csv`: Test set predictions vs. true values
- `gsnn_stratified_results.csv`: Performance metrics stratified by drug

## train_nn.py

This script trains a standard feed-forward Neural Network as a baseline comparison to GSNN.

### Usage

Basic usage with default parameters:
```bash
python train_nn.py
```

With custom hyperparameters:
```bash
python train_nn.py \
    --hidden-channels 512 \
    --layers 3 \
    --lr 1e-4 \
    --dropout 0.1 \
    --batch-size 128 \
    --epochs 50 \
    --output-dir ../results/nn_experiment
```

### Key Parameters

**Model Architecture:**
- `--hidden-channels`: Hidden units per layer (default: 1024)
- `--layers`: Number of hidden layers (default: 4)
- `--dropout`: Dropout probability (default: 0.05)
- `--nonlin`: Activation function (ELU, ReLU, LeakyReLU, GELU, default: ELU)
- `--norm`: Normalization method (batch, layer, none, default: batch)

**Training Configuration:**
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay for regularization (default: 1e-2)
- `--batch-size`: Batch size (default: 256)
- `--epochs`: Maximum epochs (default: 100)
- `--patience`: Early stopping patience (default: 10)

### Outputs

The script generates the following files in the output directory:
- `nn_model.pt`: Trained NN model
- `nn_predictions.csv`: Test set predictions vs. true values
- `nn_stratified_results.csv`: Performance metrics stratified by drug

## Complete Workflow Example

Here's an example of running the complete pipeline:

```bash
# Step 1: Construct the biological graph
python make_graph.py \
    --resp-norm zscore \
    --expr-norm unit_norm \
    --output-dir ../proc \
    --seed 42

# Step 2: Train GSNN model
python train_gsnn.py \
    --data-dir ../proc \
    --output-dir ../results/gsnn \
    --seed 42

# Step 3: Train NN baseline
python train_nn.py \
    --data-dir ../proc \
    --output-dir ../results/nn \
    --seed 42
```

## Performance Comparison

Both training scripts output comprehensive evaluation metrics including:
- Overall test performance (R², Pearson R, Spearman R, MSE)
- Stratified performance by individual drugs
- Training time and model complexity statistics

The stratified results files can be used to compare GSNN vs. NN performance across different drugs and identify where biological graph structure provides the most benefit.

### Data Requirements

The script expects the following input files (paths configurable via arguments):
- Drug response data (BeatAML format)
- Clinical mapping data
- Drug-target interaction data (Targetome)
- Drug metadata with InChI keys
- Reactome pathway annotations
- Gene expression data

Run `python make_graph.py --help` for full parameter documentation. 