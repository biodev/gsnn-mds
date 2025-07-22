import numpy as np 
import pandas as pd 
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

def stratified_drug_evaluation(x, y, yhat, node_names_dict):
    """
    Perform stratified evaluation of predictions within each drug or drug combination.
    Returns a pandas DataFrame with results.
    
    Parameters:
    -----------
    x : torch.Tensor or np.ndarray
        Input tensor of shape (n_samples, n_features)
    y : torch.Tensor or np.ndarray
        True response values
    yhat : torch.Tensor or np.ndarray
        Predicted response values
    node_names_dict : dict
        Dictionary containing node names, with 'input' key containing input node names
    
    Returns:
    --------
    pd.DataFrame : DataFrame with drug/combination names as index and evaluation metrics as columns
    """
    # Convert to numpy if needed
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    if hasattr(y, 'cpu'):
        y = y.cpu().numpy()
    if hasattr(yhat, 'cpu'):
        yhat = yhat.cpu().numpy()
    
    # Get drug node indices and names
    input_names = node_names_dict['input']
    drug_indices = []
    drug_names = []
    
    for i, name in enumerate(input_names):
        if name.startswith('DRUG__'):
            drug_indices.append(i)
            drug_names.append(name.replace('DRUG__', ''))
    
    # Create a mapping from sample index to drug combination
    sample_to_combination = {}
    for sample_idx in range(x.shape[0]):
        # Find all active drugs (value = 1.0) for this sample
        active_drugs = []
        for drug_idx, drug_name in zip(drug_indices, drug_names):
            if x[sample_idx, drug_idx] == 1.0:
                active_drugs.append(drug_name)
        
        # Sort drugs to ensure consistent naming for combinations
        active_drugs.sort()
        
        if len(active_drugs) == 1:
            combination_name = active_drugs[0]
        elif len(active_drugs) > 1:
            combination_name = ' + '.join(active_drugs)
        else:
            continue  # Skip samples with no drugs
        
        sample_to_combination[sample_idx] = combination_name
    
    # Group samples by drug combination
    combination_groups = {}
    for sample_idx, combination_name in sample_to_combination.items():
        if combination_name not in combination_groups:
            combination_groups[combination_name] = {'y': [], 'yhat': []}
        combination_groups[combination_name]['y'].append(y[sample_idx])
        combination_groups[combination_name]['yhat'].append(yhat[sample_idx])
    
    # Calculate metrics for each combination
    results = []
    for combination_name, group_data in combination_groups.items():
        y_combo = np.array(group_data['y'])
        yhat_combo = np.array(group_data['yhat'])
        
        row = {'drug_combination': combination_name, 'n_samples': len(y_combo)}
        
        if len(y_combo) > 1:  # Need at least 2 samples for correlation metrics
            row['r2'] = r2_score(y_combo, yhat_combo)
            row['pearson_r'] = np.corrcoef(y_combo, yhat_combo)[0, 1]
            row['mse'] = np.mean((y_combo - yhat_combo) ** 2)
            
            # Spearman correlation
            if len(np.unique(y_combo)) > 1:  # Need variation in y values
                row['spearman_r'] = spearmanr(y_combo, yhat_combo).correlation
            else:
                row['spearman_r'] = np.nan
        else:
            row['r2'] = np.nan
            row['pearson_r'] = np.nan
            row['mse'] = np.mean((y_combo - yhat_combo) ** 2) if len(y_combo) > 0 else np.nan
            row['spearman_r'] = np.nan
        
        results.append(row)
    
    # Add overall statistics
    overall_row = {
        'drug_combination': 'overall',
        'n_samples': len(y),
        'r2': r2_score(y, yhat),
        'pearson_r': np.corrcoef(y, yhat)[0, 1],
        'mse': np.mean((y - yhat) ** 2),
        'spearman_r': spearmanr(y, yhat).correlation
    }
    results.append(overall_row)
    
    # Convert to DataFrame and set index
    df = pd.DataFrame(results)
    df = df.set_index('drug_combination')
    
    return df
