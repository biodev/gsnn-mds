#!/usr/bin/env python3
"""
GSNN Training Script for Drug Response Prediction

This script trains a Graph Structured Neural Network (GSNN) on the biological graph
constructed by make_graph.py. The model learns to predict drug response based on 
gene expression profiles and biological pathway knowledge.

Key features:
- Configurable hyperparameters via command line arguments
- Early stopping based on validation loss
- Comprehensive evaluation including stratified drug analysis
- Model checkpointing and results saving
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from pathlib import Path
import time
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from gsnn.models.GSNN import GSNN
from gsnn_mds.data.AMLDataset import AMLDataset
from gsnn_mds.eval.stratified_drug_eval import stratified_drug_evaluation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GSNN for drug response prediction')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='../proc',
                       help='Directory containing processed data files')
    parser.add_argument('--output-dir', type=str, default='../results',
                       help='Output directory for model and results')
    
    # Model hyperparameters
    parser.add_argument('--channels', type=int, default=5,
                       help='Number of channels in GSNN layers')
    parser.add_argument('--layers', type=int, default=8,
                       help='Number of GSNN layers')
    parser.add_argument('--dropout', type=float, default=0.05,
                       help='Dropout probability')
    parser.add_argument('--nonlin', type=str, default='ELU',
                       choices=['ELU', 'ReLU', 'LeakyReLU', 'GELU'],
                       help='Non-linear activation function')
    parser.add_argument('--bias', action='store_true', default=True,
                       help='Use bias in linear layers')
    parser.add_argument('--node-attn', action='store_true', default=True,
                       help='Use node attention mechanism')
    parser.add_argument('--share-layers', action='store_false', default=False,
                       help='Share weights across GSNN layers')
    parser.add_argument('--add-function-self-edges', action='store_true', default=True,
                       help='Add self-edges to function nodes')
    parser.add_argument('--norm', type=str, default='batch',
                       choices=['batch', 'layer', 'none'],
                       help='Normalization method')
    parser.add_argument('--init', type=str, default='xavier_normal',
                       choices=['xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'],
                       help='Weight initialization method')
    parser.add_argument('--residual', action='store_true', default=True,
                       help='Use residual connections')
    parser.add_argument('--checkpoint', action='store_true', default=True,
                       help='Use gradient checkpointing to save memory')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                       help='Weight decay for regularization')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                       help='Minimum improvement for early stopping')
    
    # System settings
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Evaluation settings
    parser.add_argument('--save-predictions', action='store_true', default=True,
                       help='Save test predictions and targets')
    parser.add_argument('--stratified-eval', action='store_true', default=True,
                       help='Perform stratified evaluation by drug')
    
    return parser.parse_args()


def get_nonlin_function(nonlin_name):
    """Get the non-linear activation function."""
    nonlin_map = {
        'ELU': torch.nn.ELU,
        'ReLU': torch.nn.ReLU,
        'LeakyReLU': torch.nn.LeakyReLU,
        'GELU': torch.nn.GELU,
    }
    return nonlin_map[nonlin_name]


def get_norm_function(norm_name):
    """Get the normalization function."""
    if norm_name == 'batch':
        return 'batch'
    elif norm_name == 'layer':
        return torch.nn.LayerNorm
    elif norm_name == 'none':
        return None
    else:
        raise ValueError(f"Unknown normalization: {norm_name}")


def load_data(data_dir):
    """Load the processed data files."""
    print("Loading processed data...")
    
    data_dir = Path(data_dir)
    
    # Load graph structure
    data = torch.load(data_dir / 'graph.pt', weights_only=False)
    
    # Load expression data
    aml_expr = pd.read_csv(data_dir / 'aml_expr.csv', index_col=0)
    aml_expr = aml_expr.fillna(0)
    
    # Load response data
    drug = pd.read_csv(data_dir / 'resp.csv')
    
    print(f"Loaded graph with {len(data.node_names_dict['input'])} input nodes")
    print(f"Loaded expression data: {aml_expr.shape[0]} samples x {aml_expr.shape[1]} genes")
    print(f"Loaded response data: {len(drug)} measurements")
    
    return data, aml_expr, drug


def create_input_mapping(aml_expr, data):
    """Create mapping from patient IDs to input vectors."""
    print("Creating input mapping...")
    
    id2x = {}
    expr_ixs = np.array([i for i, n in enumerate(data.node_names_dict['input']) if "EXPR__" in n])
    expr_names = np.array(data.node_names_dict['input'])[expr_ixs]
    
    for i, row in aml_expr.iterrows():
        x = torch.zeros(len(data.node_names_dict['input']), dtype=torch.float32)
        x[expr_ixs] = torch.tensor(row[expr_names].values.astype(np.float32), dtype=torch.float32)
        id2x[row.name] = x  # row.name is the index (patient ID)
    
    print(f"Created input mapping for {len(id2x)} patients")
    return id2x


def create_data_loaders(id2x, drug, data, batch_size, num_workers):
    """Create data loaders for train/val/test splits."""
    print("Creating data loaders...")
    
    input_names = data.node_names_dict['input']
    
    train_dataset = AMLDataset(id2x, drug[drug.partition == 'train'], input_names)
    val_dataset = AMLDataset(id2x, drug[drug.partition == 'val'], input_names)
    test_dataset = AMLDataset(id2x, drug[drug.partition == 'test'], input_names)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def create_model(data, args, device):
    """Create and initialize the GSNN model."""
    print("Creating GSNN model...")
    
    # Get activation and normalization functions
    nonlin = get_nonlin_function(args.nonlin)
    norm = get_norm_function(args.norm)
    
    model = GSNN(
        edge_index_dict=data.edge_index_dict,
        node_names_dict=data.node_names_dict,
        channels=args.channels,
        layers=args.layers,
        dropout=args.dropout,
        nonlin=nonlin,
        bias=args.bias,
        node_attn=args.node_attn,
        share_layers=args.share_layers,
        add_function_self_edges=args.add_function_self_edges,
        norm=norm,
        init=args.init,
        residual=args.residual,
        checkpoint=args.checkpoint
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created model with {num_params:,} trainable parameters")
    
    return model


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_r2 = 0.0
    num_batches = 0
    
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        
        yhat = model(x.to(device))
        loss = criterion(yhat.squeeze(), y.squeeze().to(device))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        r2 = r2_score(y.detach().numpy(), yhat.detach().cpu().numpy())
        total_r2 += r2
        num_batches += 1
        
        print(f'[{i+1}/{len(train_loader)} Loss: {loss.item():.4f} | R2: {r2:.4f}]', end='\r')
    
    avg_loss = total_loss / num_batches
    avg_r2 = total_r2 / num_batches
    
    return avg_loss, avg_r2


def validate_model(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    all_y = []
    all_yhat = []
    
    with torch.no_grad():
        for x, y in val_loader:
            yhat = model(x.to(device))
            all_y.append(y.squeeze().numpy())
            all_yhat.append(yhat.squeeze().detach().cpu().numpy())
    
    all_y = np.concatenate(all_y)
    all_yhat = np.concatenate(all_yhat)
    
    mse = np.mean((all_y - all_yhat) ** 2)
    r2 = r2_score(all_y, all_yhat)
    spearman_r = spearmanr(all_y, all_yhat).correlation
    
    return mse, r2, spearman_r


def test_model(model, test_loader, device):
    """Test the model and return predictions."""
    model.eval()
    all_x = []
    all_y = []
    all_yhat = []
    
    with torch.no_grad():
        for x, y in test_loader:
            yhat = model(x.to(device))
            all_x.append(x.squeeze().detach().cpu().numpy())
            all_y.append(y.squeeze().detach().numpy())
            all_yhat.append(yhat.squeeze().detach().cpu().numpy())
    
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y)
    all_yhat = np.concatenate(all_yhat)
    
    return all_x, all_y, all_yhat


def save_results(model, all_x, all_y, all_yhat, data, args, output_dir):
    """Save model and results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model, output_dir / 'gsnn_model.pt')
    print(f"Model saved to: {output_dir / 'gsnn_model.pt'}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            'y_true': all_y,
            'y_pred': all_yhat
        })
        predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
        print(f"Predictions saved to: {output_dir / 'predictions.csv'}")
    
    # Perform stratified evaluation if requested
    if args.stratified_eval:
        stratified_results = stratified_drug_evaluation(all_x, all_y, all_yhat, data.node_names_dict)
        stratified_results.to_csv(output_dir / 'stratified_results.csv')
        print(f"Stratified results saved to: {output_dir / 'stratified_results.csv'}")
        return stratified_results
    
    return None


def print_training_summary(args, total_time):
    """Print comprehensive training summary."""
    print("\n" + "="*60)
    print("GSNN TRAINING SUMMARY")
    print("="*60)
    
    print(f"\n🏗️ MODEL ARCHITECTURE:")
    print(f"   Channels: {args.channels}")
    print(f"   Layers: {args.layers}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Activation: {args.nonlin}")
    print(f"   Normalization: {args.norm}")
    print(f"   Residual connections: {args.residual}")
    print(f"   Node attention: {args.node_attn}")
    
    print(f"\n⚙️ TRAINING CONFIGURATION:")
    print(f"   Learning rate: {args.lr}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max epochs: {args.epochs}")
    print(f"   Device: {args.device}")
    
    print(f"\n⏱️ TRAINING TIME:")
    print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    print("="*60)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=== GSNN Training ===")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    
    start_time = time.time()
    
    # Load data
    data, aml_expr, drug = load_data(args.data_dir)
    
    # Create input mapping
    id2x = create_input_mapping(aml_expr, data)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        id2x, drug, data, args.batch_size, args.num_workers
    )
    
    # Create model
    model = create_model(data, args, device)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_r2 = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_r2, val_spearman = validate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | Train R2: {train_r2:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val R2: {val_r2:.4f} | '
              f'Val Spearman: {val_spearman:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    
    # Test model
    print("\nEvaluating on test set...")
    all_x, all_y, all_yhat = test_model(model, test_loader, device)
    
    # Calculate test metrics
    test_mse = np.mean((all_y - all_yhat) ** 2)
    test_r2 = r2_score(all_y, all_yhat)
    test_pearson = np.corrcoef(all_y, all_yhat)[0, 1]
    test_spearman = spearmanr(all_y, all_yhat).correlation
    
    print(f"\n🎯 TEST RESULTS:")
    print(f"   MSE: {test_mse:.4f}")
    print(f"   R²: {test_r2:.4f}")
    print(f"   Pearson R: {test_pearson:.4f}")
    print(f"   Spearman R: {test_spearman:.4f}")
    
    # Save results
    stratified_results = save_results(model, all_x, all_y, all_yhat, data, args, args.output_dir)
    
    total_time = time.time() - start_time
    print_training_summary(args, total_time)
    
    if stratified_results is not None:
        print(f"\n📊 STRATIFIED EVALUATION:")
        print(f"   Number of drugs evaluated: {len(stratified_results) - 1}")  # -1 for 'overall'
        overall_stats = stratified_results.loc['overall']
        print(f"   Overall R²: {overall_stats['r2']:.4f}")
        print(f"   Overall Spearman R: {overall_stats['spearman_r']:.4f}")
    
    print("\n=== Training Complete ===")


if __name__ == '__main__':
    main() 