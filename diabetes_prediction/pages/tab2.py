"""Tab page for streamlit app
"""

from diabetes_prediction.utils.app.io import *
from diabetes_prediction.utils.data.dataloader import *
from diabetes_prediction.utils.data.preprocessing import *


def _compare_datas(before, after, meta, cols, title):
    df_desc = meta[meta['final_id'].isin(cols)].drop(columns=['question_id', 'keywords'])

    fig, axes = plt.subplots(2, 5, figsize=PARAMS.figsize)
    fig.suptitle(title, fontsize=15)
    for axes_row, df in zip(axes, [before, after]):
        for ax, col in zip(axes_row.flat, df_desc['final_id']):
            if col in df.select_dtypes('number'):
                sns.histplot(df[col], kde=True, bins=100, ax=ax)
            else:
                sns.countplot(df, x=col, ax=ax)
            if ax != axes_row[0]:
                ax.set_ylabel(None)
    for ax in axes[0]:
        ax.set_xlabel(None)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

def show_replace_ambiguous_options(metas, datas, data_id):
    meta, data = metas[data_id], datas[data_id]

    before = data.copy()
    replace_ambiguous_options(data, meta)
    after = data.copy()

    diff = (before.nunique() - after.nunique()) > 0
    cols = diff[diff].index[:5]

    _compare_datas(before, after, meta, cols, "Number of options for each features (up: before / down: after)")

def show_impute_data(metas, datas, data_id):
    meta, data = metas[data_id], datas[data_id]

    before = data.copy()
    impute_data(data, meta)
    after = data.copy()

    idxs = before.isna().any()
    cols = np.random.choice(idxs[idxs].index, 5, replace=False)

    _compare_datas(before, after, meta, cols, "Impute data (up: before / down: after)")

def show_set_dtypes(metas, datas, data_id):
    meta, data = metas[data_id], datas[data_id]
    set_dtypes(data, meta)

    show_numerical_features(metas, datas, data_id)
    show_categorical_features(metas, datas, data_id)

def show_impute_numerical_features(metas, datas, data_id):
    meta, data = metas[data_id], datas[data_id]

    before = data.copy()
    impute_numerical_features(data, meta)
    after  = data

    num_cols = data.select_dtypes('number').columns
    cols = (before[num_cols] == -1).sum().sort_values(ascending=False).index[:5]  # most effective columns
    _compare_datas(before, after, meta, cols, "Impute numerical data (up: before / down: after)")

def show_extract_features(metas, datas, data_id, target=None):
    meta, data = metas[data_id], datas[data_id]
    extract_features(data, target)

    with st_stdout("code"):
        print("family_id: HHX + FMX + SRVY_YR")
        if target:
            print(f"label: {target}")
            print(f"\t0: {target} = 'no'")
            print(f"\t1: {target} = 'yes'")
            print(f"\t2: {target} = 'unknown'")


def show_drop_columns(metas, datas, data_id):
    meta, data = metas[data_id], datas[data_id]

    before = data.copy()
    drop_columns(data)

    st.write("- Drop constant columns")
    cnts = before.nunique().sort_values()
    constant_cols = cnts[cnts == 1].index
    st.dataframe(before[constant_cols].nunique())

    st.write("- Drop redundant columns")
    redundant_cols = ['HHX', 'FMX', 'INTV_QRT']
    cols = [col for col in redundant_cols if col in before]
    st.dataframe(before[cols].nunique())


def show_manual_handling(metas, datas, data_id):
    meta, data = metas[data_id], datas[data_id]

    before = data.copy()
    manual_handling(data, meta)

    st.write("- Handle **'do nothing'** options")
    cols = []
    for col in before:
        if col in ('label', 'family_id'):
            continue
        options = get_meta_values(meta, col)['options']
        options_inversed = inverse_dict(options)
        for option in options_inversed:
            if re.search("^unable to do ", option) or (option == 'never'):
                cols.append(col)

    _compare_datas(before, data, meta, cols, "Manual handling")


def show_drop_diabetes_columns(metas, datas, data_id):
    meta, data = metas[data_id], datas[data_id]

    idxs = meta['keywords'].astype(str).str.contains('diabetes')
    diabetes_cols = meta.loc[idxs[idxs].index, 'final_id']
    diabetes_cols = [col for col in diabetes_cols if col in data]
    drop_diabetes_columns(data, meta)

    st.write(f"- Drop columns")
    st.dataframe(meta[meta['final_id'].isin(diabetes_cols)][['final_id', 'description', 'options', 'keywords']])

def show_correlations():
    metas, datas = load_dataset()
    family       = pd.read_feather(PATH.family_final_data)
    sample_adult = pd.read_feather(PATH.sample_adult_final_data)
    sample_adult_family = pd.merge(sample_adult, family, how='left', on='family_id')

    data = sample_adult_family
    meta = pd.concat([metas['family'], metas['sample_adult']])
    corr = get_corr(data)

    n_cols = 30
    df_desc = get_description(meta, corr.index[:n_cols])
    fig, ax = plt.subplots(figsize=(30, 3))
    corr['corr_abs'][:n_cols].plot.bar(title=f"abs(Correlation Coefficient) (top {n_cols} features from {len(corr)} features)", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)
