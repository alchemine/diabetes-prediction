"""Metrics module
"""
import pandas as pd

from diabetes_prediction._utils import *

from statsmodels.stats.outliers_influence import variance_inflation_factor
from dask import delayed, compute
from dask.diagnostics import ProgressBar


@T
def get_corr(metadata: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Get correlation of `data` with target column.

    Args:
        metadata: Metadata dataframe.
        data: Dataframe.

    Returns:
        Correlation dataframe.
    """
    target = PARAMS.target
    assert target in data, f"Data should have {target} column"

    # Get correlation
    C = data.select_dtypes('number').corr()
    corr = pd.DataFrame([C[target], C[target].abs()], index=['corr', 'corr_abs']).T
    corr.index.name = 'final_id'
    corr.sort_values('corr_abs', ascending=False, inplace=True)
    corr = corr.iloc[1:]

    # Merge with metadata
    corr = pd.merge(corr, metadata.set_index('final_id'), how='left', on='final_id')
    corr = corr.sort_values('corr_abs', ascending=False)
    return corr[['corr_abs', 'description', 'options']]


@T
def get_VIF(data: pd.DataFrame, plot: bool = False) -> pd.DataFrame:
    """Get variance inflation factor.

    Args:
        data: Dataframe.
        plot: Whether to plot or not.

    Returns:
        Variance inflation factor.
    """
    num_data = data.select_dtypes('number')
    if PARAMS.target in num_data:
        num_data.drop(columns=PARAMS.target, inplace=True)

    rst = pd.DataFrame(index=num_data.columns)
    with ProgressBar():
        tasks = [delayed(variance_inflation_factor)(num_data.values, i) for i in range(len(num_data.columns))]
        rst['VIF'] = compute(*tasks, scheduler='processes')

    if plot:
        rst.plot.bar(figsize=PARAMS.figsize)
    return rst


def plot_correlations(data, vif_data, cols):
    data = data[cols]
    corr = data.corr()

    fig, axes = plt.subplots(2, figsize=(28, 12))

#     mask = np.zeros_like(corr)
#     mask[np.triu_indices_from(mask)] = True
#     sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.1f', center=0, ax=axes[0], cbar=False)

    axes[0].set_title("Correlation Matrix (C ≥ 0.25)")
    mask = np.zeros_like(corr)
    mask[(corr < 0.25) | (corr == 1)] = True
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.1f', center=0, ax=axes[0], cbar=False)

    vif_data.plot.bar(ax=axes[1], ylabel='VIF')
    axes[1].axhline(5, color='k')
