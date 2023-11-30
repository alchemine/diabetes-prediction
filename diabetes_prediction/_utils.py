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
