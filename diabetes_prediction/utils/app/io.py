"""Streamlit I/O module
"""

from diabetes_prediction._utils import *

from io import StringIO
from threading import current_thread
from contextlib import contextmanager

import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME


def run_command(*args, **kwargs):
    result = subprocess.Popen(stdout=subprocess.PIPE, *args, **kwargs)
    for line in result.stdout:
        print(line.decode('utf-8'))


# Reference: https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602/9
@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield

# @contextmanager
# def st_stderr(dst):
#     with st_redirect(sys.stderr, dst):
#         yield


def show_numerical_features(metas, datas, data_id):
    meta, data = metas[data_id], datas[data_id]

    data_num = data.select_dtypes('number')
    n_features, n_features_sample = len(data_num.columns), 5
    n_rows, n_samples = len(data_num), 5000

    cols    = np.random.choice(data_num.columns, n_features_sample, replace=False)
    sample  = data_num.sample(n_samples)[cols]
    df_desc = meta[meta['final_id'].isin(cols)].drop(columns='question_id')

    title = f"Numerical Features (sampling: {n_features_sample} features from {n_features} features, {n_samples} rows from {n_rows} rows)"
    fig, axes = plt.subplots(1, n_features_sample, figsize=PARAMS.figsize)
    for idx, ax in enumerate(axes.flat):
        col = df_desc['final_id'].values[idx]
        sns.histplot(sample[col], kde=True, bins=100, ax=ax)
        if ax != axes.flat[0]:
            ax.set_ylabel(None)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

def show_categorical_features(metas, datas, data_id):
    meta, data = metas[data_id], datas[data_id]

    data_cat = data.select_dtypes('object')
    n_features, n_features_sample = len(data_cat.columns), 5
    n_rows, n_samples = len(data_cat), 5000

    cols    = np.random.choice(data_cat.columns, n_features_sample, replace=False)
    sample  = data_cat.sample(n_samples)[cols]
    df_desc = meta[meta['final_id'].isin(cols)].drop(columns='question_id')

    title = f"Categorical Features (sampling: {n_features_sample} features from {n_features} features, {n_samples} rows from {n_rows} rows)"
    fig, axes = plt.subplots(1, n_features_sample, figsize=PARAMS.figsize)
    for idx, ax in enumerate(axes.flat):
        col = df_desc['final_id'].values[idx]
        sns.countplot(data=sample, x=col, ax=ax)
        if ax != axes.flat[0]:
            ax.set_ylabel(None)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

def get_description(meta, final_ids):
    final_ids = [id for id in final_ids if id in meta['final_id'].values]
    return meta.set_index('final_id').loc[final_ids][['description', 'options', 'keywords']]

def get_corr(data):
    assert 'label' in data, "Data should have 'label' column"

    C = data.select_dtypes('number').corr()
    corr = pd.DataFrame([C['label'], C['label'].abs()], index=['corr', 'corr_abs']).T
    corr.sort_values('corr_abs', ascending=False, inplace=True)
    corr = corr.iloc[1:]
    corr['importance'] = corr['corr_abs'].cumsum()/corr['corr_abs'].cumsum().sum()
    return corr

