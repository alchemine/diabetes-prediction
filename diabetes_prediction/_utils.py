"""Utility module
"""

from diabetes_prediction.common import *


def get_meta_values(meta, col):
    meta_col = meta[meta['final_id'] == col]
    if len(meta_col) == 0:
        raise ValueError(f"[FAILURE] There is no '{col}' column in meta data.")
    rst = meta_col.to_dict(orient='records')[0]

    # Append 'unknown' option
    #   See details in `diabetes_prediction.utils.data.preprocessing.replace_ambiguous_options()`
    if 'unknown' not in inverse_dict(rst['options']):
        rst['options']['-1'] = 'unknown'

    return rst


def get_unknown_value(meta, col):
    options_inversed = inverse_dict(get_meta_values(meta, col)['options'])
    return options_inversed['unknown']


def load_metadata(path: str) -> pd.DataFrame:
    """Load metadata and evaluate columns (dictionary or list).

    Args:
        path: Path to metadata.

    Returns:
        Metadata.
    """
    metadata = pd.read_csv(path)
    for col in ('options', 'keywords'):
        metadata[col] = metadata[col].map(eval)
    return metadata


def extract_family_id(data: pd.DataFrame) -> None:
    """Extract and append column `family_id` using `HHX`, `FMX`, `SRVY_YR`.
    FMX: Use this variable in combination with HHX and SRVY_YR(constant, 2018) to identify individual families.

    Args:
        data: Data records.
    """
    data['family_id'] = data['HHX'] + data['FMX'] + data['SRVY_YR']


def merge_features_metadata(features: pd.Index, metadata: pd.DataFrame) -> pd.DataFrame:
    """Merge features and metadata.

    Args:
        features: Features.
        metadata: Metadata.

    Returns:
        Data records with metadata.
    """
    return pd.merge(pd.DataFrame(features, columns=['final_id']), metadata, how='left', on='final_id')
