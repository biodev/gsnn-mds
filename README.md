# Drug Response Prediction Pipeline

A Snakemake pipeline for predicting drug response using Graph Structured Neural Networks (GSNN) and biological pathway knowledge.

## Overview

This pipeline implements a comprehensive drug response prediction workflow that:

1. **Graph Construction** (`make_graph.py`): Processes biological interaction networks, drug-target interactions, and gene expression data to create a heterogeneous graph
2. **Neural Network Training** (`train_nn.py`): Trains a standard feed-forward neural network as a baseline
3. **GSNN Training** (`train_gsnn.py`): Trains a Graph Structured Neural Network that incorporates biological pathway structure

## Project Structure

```
├── workflow/
│   └── Snakefile              # Main pipeline definition
├── config/
│   ├── config.yaml            # Main configuration
│   └── config_test.yaml       # Test configuration (smaller/faster)
├── scripts/
│   ├── make_graph.py          # Graph construction
│   ├── train_nn.py            # Neural network training
│   └── train_gsnn.py          # GSNN training
├── exp/
│   └── {config_name}/         # Experiment outputs (created by pipeline)
│       ├── graph/             # Graph construction outputs
│       ├── nn/                # Neural network results
│       └── gsnn/              # GSNN results
├── run_pipeline.sh            # Helper script for easy pipeline execution
└── README.md
```

## Pipeline Structure

```
Input Data
    ↓
Graph Construction (make_graph) → exp/{config_name}/graph/
    ↓
    ├── Neural Network Training (train_nn) → exp/{config_name}/nn/
    └── GSNN Training (train_gsnn) → exp/{config_name}/gsnn/
```

## Installation & Requirements

### Dependencies
- Python 3.8+
- Snakemake 7.0+
- PyTorch
- pandas
- numpy
- networkx
- scikit-learn
- scipy
- pypath
- Custom modules: `gsnn`, `gsnn_mds`, `lincs_gsnn`

### Setup
```bash
# Clone/download the repository
git clone <repository_url>
cd drug-response-pipeline

# Create conda environment (optional but recommended)
conda create -n drug-response python=3.8
conda activate drug-response

# Install dependencies
pip install snakemake pandas numpy torch networkx scikit-learn scipy
# Install custom modules as needed
```

## Configuration

All pipeline parameters are controlled through configuration files in the `config/` directory:
- `config/config.yaml`: Main configuration for production runs
- `config/config_test.yaml`: Test configuration with smaller parameters for development

### Key Configuration Sections

#### Config Name (`config_name`)
- Unique identifier for the experiment (determines output directory name)
- Example: `config_name: "main_experiment"` creates outputs in `exp/main_experiment/`

#### Data Paths (`data`)
- `drug_data`: Drug response measurements
- `clinical_data`: Clinical annotations and sample mappings
- `targetome_data`: Drug-target interaction data
- `drug_meta`: Drug metadata with InChI keys
- `reactome_data`: Pathway annotations
- `expression_data`: Gene expression profiles

#### Graph Construction (`graph`)
- `resp_norm`: Response normalization method (zscore, unit_norm, within_drug_zscore, none)
- `graph_depth`: Maximum depth for graph subsetting
- `expr_var_quantile`: Minimum variance quantile for gene filtering
- `expr_norm`: Expression normalization (unit_norm, zscore)
- `min_pathway_size`: Minimum proteins per pathway
- `max_assay_value`: Maximum Ki/Kd value for drug-target interactions
- `train_frac`: Training fraction (test uses MDS logic)

#### Neural Network (`nn`)
- `hidden_channels`: Hidden layer size
- `layers`: Number of hidden layers
- `dropout`: Dropout probability
- `lr`: Learning rate
- `batch_size`: Training batch size
- `epochs`: Maximum training epochs

#### GSNN (`gsnn`)
- `channels`: GSNN layer channels
- `layers`: Number of GSNN layers
- `node_attn`: Enable node attention
- `residual`: Enable residual connections
- `checkpoint`: Enable gradient checkpointing

## Usage

### Basic Pipeline Execution

#### Option 1: Using the Helper Script (Recommended)

```bash
# Run the complete pipeline with default configuration
./run_pipeline.sh

# Run test configuration for faster execution
./run_pipeline.sh --config config/config_test.yaml --cores 4

# Dry run to see planned execution
./run_pipeline.sh --dry-run

# Run only graph construction
./run_pipeline.sh --target exp/main_experiment/graph/graph.pt

# Run on cluster
./run_pipeline.sh --cluster "sbatch --cpus-per-task={threads} --mem={resources.mem_mb}M" --jobs 5

# Get help with all options
./run_pipeline.sh --help
```

#### Option 2: Direct Snakemake Commands

```bash
# Run the complete pipeline (using default config/config.yaml)
snakemake --snakefile workflow/Snakefile --cores 8

# Run with test configuration for faster execution
snakemake --snakefile workflow/Snakefile --configfile config/config_test.yaml --cores 4

# Dry run to see planned execution
snakemake --snakefile workflow/Snakefile --dry-run

# Run only neural network training  
snakemake --snakefile workflow/Snakefile exp/main_experiment/nn/nn_model.pt

# Run with custom config name
snakemake --snakefile workflow/Snakefile --config config_name=my_experiment --cores 8
```

### Cluster Execution

```bash
# SLURM cluster example
snakemake --snakefile workflow/Snakefile \
  --cluster "sbatch --cpus-per-task={threads} --mem={resources.mem_mb} --time={resources.runtime}" \
  --jobs 10

# With GPU support
snakemake --snakefile workflow/Snakefile \
  --cluster "sbatch --cpus-per-task={threads} --mem={resources.mem_mb} --time={resources.runtime} --gres=gpu:{resources.gpu}" \
  --jobs 5

# Test run on cluster
snakemake --snakefile workflow/Snakefile --configfile config/config_test.yaml \
  --cluster "sbatch --cpus-per-task={threads} --mem={resources.mem_mb} --time={resources.runtime}" \
  --jobs 3
```

### Resource Management

Configure resources in `config.yaml`:
```yaml
resources:
  make_graph:
    memory: 16000  # MB
    runtime: 180   # minutes
  train_nn:
    memory: 8000   # MB
    runtime: 120   # minutes
    gpu: 1
  train_gsnn:
    memory: 12000  # MB
    runtime: 180   # minutes
    gpu: 1
```

## Outputs

### Directory Structure
```
exp/
└── {config_name}/               # Experiment identifier (e.g., "main_experiment")
    ├── graph/                   # Graph construction outputs
    │   ├── graph.pt            # PyTorch Geometric heterogeneous graph
    │   ├── aml_expr.csv        # Processed expression data
    │   └── resp.csv            # Drug response data with partitions
    ├── nn/                      # Neural network training outputs
    │   ├── nn_model.pt         # Trained neural network model
    │   ├── predictions.csv     # NN test predictions
    │   └── stratified_results.csv # NN drug-stratified evaluation
    └── gsnn/                    # GSNN training outputs
        ├── gsnn_model.pt       # Trained GSNN model
        ├── predictions.csv     # GSNN test predictions
        └── stratified_results.csv # GSNN drug-stratified evaluation
```

### Multiple Experiments
You can run multiple experiments with different configurations:
```bash
# Experiment 1: Main configuration
snakemake --snakefile workflow/Snakefile --cores 8
# Outputs: exp/main_experiment/

# Experiment 2: Test configuration
snakemake --snakefile workflow/Snakefile --configfile config/config_test.yaml --cores 4
# Outputs: exp/test_experiment/

# Experiment 3: Custom configuration
snakemake --snakefile workflow/Snakefile --config config_name=hyperopt_run --cores 8
# Outputs: exp/hyperopt_run/
```

### Key Output Files

- **Graph Structure** (`graph.pt`): PyTorch Geometric heterogeneous graph with biological interactions
- **Models** (`*_model.pt`): Trained PyTorch models ready for inference
- **Predictions** (`*_predictions.csv`): Test set predictions with true/predicted values
- **Stratified Results** (`*_stratified_results.csv`): Performance metrics stratified by individual drugs

## Data Partitioning

The pipeline uses a specialized partitioning strategy:
- **Training Set**: Random selection from non-MDS patients (default: 85%)
- **Validation Set**: Remaining non-MDS patients (default: 15%)
- **Test Set**: All patients with prior MDS diagnosis

This ensures the test set represents a clinically relevant population for model evaluation.

## Customization

### Modifying Parameters
Edit `config.yaml` to adjust:
- Data paths for your dataset
- Model hyperparameters
- Resource allocation
- Training configuration

### Adding New Rules
Extend the `Snakefile` to add:
- Additional preprocessing steps
- Different model architectures
- Custom evaluation metrics
- Ensemble methods

### Example Custom Config
```yaml
# Focus on smaller, faster models for testing
nn:
  hidden_channels: 512
  layers: 2
  epochs: 50

gsnn:
  channels: 3
  layers: 4
  epochs: 50
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Increase memory allocation in `config.yaml` resources section
2. **GPU Errors**: Set `device: "cpu"` in global config for CPU-only execution
3. **Data Path Errors**: Verify all file paths in the `data` section exist
4. **Import Errors**: Ensure all required packages and custom modules are installed

### Monitoring Progress

```bash
# Output is displayed directly in console - no separate log files
# To save output to a file if needed:
./run_pipeline.sh --config config/config_test.yaml > pipeline_output.log 2>&1

# Check Snakemake status
snakemake --snakefile workflow/Snakefile --summary
snakemake --snakefile workflow/Snakefile --list-params-changes
```

## Performance Optimization

### GPU Acceleration
- Ensure CUDA-compatible PyTorch installation
- Set `device: "cuda"` in config
- Adjust batch sizes based on GPU memory

### Memory Management
- Enable gradient checkpointing for GSNN: `checkpoint: true`
- Reduce batch sizes if experiencing OOM errors
- Use CPU for graph construction if GPU memory is limited

### Parallel Execution
- NN and GSNN training can run in parallel after graph construction
- Adjust `--jobs` parameter based on available resources
- Consider resource constraints when running multiple GPU jobs

## Citation

If you use this pipeline in your research, please cite:
- [Your paper/preprint]
- Relevant dependencies and algorithms

## License

[Specify your license here]

## Support

For questions and issues:
- Create an issue in the repository
- Output is displayed directly in console (no separate log files)
- To save output for debugging: `./run_pipeline.sh > output.log 2>&1`
- Verify configuration against the examples in `config/` directory
- Ensure you're running from the project root with `--snakefile workflow/Snakefile`