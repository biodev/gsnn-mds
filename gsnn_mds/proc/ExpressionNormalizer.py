import numpy as np
import joblib
from scipy import stats

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