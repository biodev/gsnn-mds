import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

def eval_subgroup(preds, subgroup_ids, inhibitors, subgroup_name, min_sample_size=3, test='mannwhitneyu'):
    """Compare subgroup vs. non-subgroup predicted responses for each inhibitor.

    Parameters
    ----------
    preds : pandas.DataFrame
        Prediction table with `id`, `inhibitor_1`, `inhibitor_2`, and `yhat` columns.
    subgroup_ids : collection
        Sample IDs that define the subgroup of interest.
    inhibitors : pandas.DataFrame
        Table of inhibitor combinations with `inhibitor_1` and `inhibitor_2` columns.
    subgroup_name : str
        Label to attach to the output rows.
    min_sample_size : int, default=3
        Minimum number of non-missing predictions required in each group.
    test : str, default='ttest'
        Statistical test to use. Supported values are `ttest`/`welch` and
        `mannwhitneyu`/`nonparametric`.

    Returns
    -------
    pandas.DataFrame
        One row per evaluated inhibitor with summary statistics, p-values, and
        test metadata. The `tstat` column is retained for backward compatibility
        and mirrors `test_stat`, so it is not always a t-statistic.
    """

    test_aliases = {
        'ttest': 'ttest',
        't-test': 'ttest',
        'welch': 'ttest',
        'welch_ttest': 'ttest',
        'mannwhitneyu': 'mannwhitneyu',
        'mann-whitney': 'mannwhitneyu',
        'mannwhitney': 'mannwhitneyu',
        'mw': 'mannwhitneyu',
        'nonparametric': 'mannwhitneyu',
    }
    test_method = test_aliases.get(str(test).lower())
    if test_method is None:
        raise ValueError(f"Unsupported test '{test}'. Use 'ttest' or 'mannwhitneyu'.")

    subgroup_ids = set(subgroup_ids)
    subgroup_res = {
        'drug1': [],
        'drug2': [],
        'subgroup_y_mean': [],
        'subgroup_y_std': [],
        'rest_y_mean': [],
        'rest_y_std': [],
        'pval': [],
        'test_stat': [],
        'subgroup_n': [],
        'rest_n': [],
        'test_method': [],
    }

    for ii, (_, row) in enumerate(inhibitors.reset_index(drop=True).iterrows(), start=1):
        print(f'progress: {ii}/{len(inhibitors)}', end='\r')

        tmp = preds.loc[preds.inhibitor_1 == row.inhibitor_1]
        
        if pd.isna(row.inhibitor_2):
            tmp = tmp.loc[tmp.inhibitor_2.isna()]
        else:
            tmp = tmp.loc[tmp.inhibitor_2 == row.inhibitor_2]

        subgroup_preds = tmp.loc[tmp.id.isin(subgroup_ids), 'yhat'].dropna().to_numpy()
        rest_preds = tmp.loc[~tmp.id.isin(subgroup_ids), 'yhat'].dropna().to_numpy()

        if len(subgroup_preds) < min_sample_size or len(rest_preds) < min_sample_size:
            continue

        if test_method == 'ttest':
            test_res = ttest_ind(subgroup_preds, rest_preds, equal_var=False, nan_policy='omit')
        else:
            test_res = mannwhitneyu(subgroup_preds, rest_preds, alternative='two-sided')

        subgroup_res['drug1'].append(row.inhibitor_1)
        subgroup_res['drug2'].append(row.inhibitor_2)
        subgroup_res['subgroup_y_mean'].append(subgroup_preds.mean())
        subgroup_res['subgroup_y_std'].append(subgroup_preds.std())
        subgroup_res['rest_y_mean'].append(rest_preds.mean())
        subgroup_res['rest_y_std'].append(rest_preds.std())
        subgroup_res['pval'].append(test_res.pvalue)
        subgroup_res['test_stat'].append(test_res.statistic)
        subgroup_res['subgroup_n'].append(len(subgroup_preds))
        subgroup_res['rest_n'].append(len(rest_preds))
        subgroup_res['test_method'].append(test_method)

    subgroup_res = pd.DataFrame(subgroup_res)
    subgroup_res = subgroup_res.assign(subgroup=subgroup_name)
    subgroup_res = subgroup_res.assign(mean_diff=lambda x: x.subgroup_y_mean - x.rest_y_mean)  # resistant is positive, sensitive is negative

    return subgroup_res
