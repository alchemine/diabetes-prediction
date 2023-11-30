"""Streamlit main application
"""

from diabetes_prediction.pages import *
from diabetes_prediction.utils.data.preprocessing import *


st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

st.title('Diabetes Prediction and Analysis')
tab_titles = ['Data Generating', 'Explanatory Data Analysis', 'Data Preprocessing', 'Modeling']
tabs = st.tabs(tab_titles)


with tabs[0]:
    show_generate_meta()


with tabs[1]:
    tab_subtitles = ["Family", "Sample Adult", "Sample Child"]
    subtabs       = st.tabs(tab_subtitles)

    # Common section
    metas, datas = load_dataset()
    data_ids = lmap(lambda s: s.lower().replace(' ', '_'), tab_subtitles)

    with subtabs[0]:
        data_id = data_ids[0]
        set_dtypes(datas[data_id], metas[data_id])

        show_overview(metas, datas, data_id)
        show_feature_distribution(metas, datas, data_id)

    with subtabs[1]:
        data_id = data_ids[1]
        set_dtypes(datas[data_id], metas[data_id])

        show_overview(metas, datas, data_id)
        show_feature_distribution(metas, datas, data_id)

    with subtabs[2]:
        data_id = data_ids[2]
        set_dtypes(datas[data_id], metas[data_id])

        show_overview(metas, datas, data_id)
        show_feature_distribution(metas, datas, data_id)


with tabs[2]:
    st.header("1. Replace Ambiguous Options with 'Unknown'")

    tab_subtitles = ["Family", "Sample Adult"]
    subtabs1       = st.tabs(tab_subtitles)

    # Common section
    metas, datas = load_dataset()
    data_ids = lmap(lambda s: s.lower().replace(' ', '_'), tab_subtitles)

    with subtabs1[0]:
        data_id = data_ids[0]
        show_replace_ambiguous_options(metas, datas, data_id)
    with subtabs1[1]:
        data_id = data_ids[1]
        show_replace_ambiguous_options(metas, datas, data_id)

    st.header("2. Impute Nan values to 'Unknown'(-1)")
    subtabs2 = st.tabs(tab_subtitles)
    with subtabs2[0]:
        data_id = data_ids[0]
        show_impute_data(metas, datas, data_id)
    with subtabs2[1]:
        data_id = data_ids[1]
        show_impute_data(metas, datas, data_id)

    st.header("3. Set Data types of Features (numerical or categorical)")
    subtabs3 = st.tabs(tab_subtitles)
    with subtabs3[0]:
        data_id = data_ids[0]
        show_set_dtypes(metas, datas, data_id)
    with subtabs3[1]:
        data_id = data_ids[1]
        show_set_dtypes(metas, datas, data_id)

    st.header("4. Impute Numerical Features (mode imputing)")
    subtabs4 = st.tabs(tab_subtitles)
    with subtabs4[0]:
        data_id = data_ids[0]
        show_impute_numerical_features(metas, datas, data_id)
    with subtabs4[1]:
        data_id = data_ids[1]
        show_impute_numerical_features(metas, datas, data_id)

    st.header("5. Extract Features")
    subtabs5 = st.tabs(tab_subtitles)
    with subtabs5[0]:
        data_id = data_ids[0]
        show_extract_features(metas, datas, data_id)
    with subtabs5[1]:
        data_id = data_ids[1]
        show_extract_features(metas, datas, data_id, target='DIBEV1')

    st.header("6. Drop Columns")
    subtabs6 = st.tabs(tab_subtitles)
    with subtabs6[0]:
        data_id = data_ids[0]
        meta, data = metas[data_id], datas[data_id]
        show_drop_columns(metas, datas, data_id)
    with subtabs6[1]:
        data_id = data_ids[1]
        show_drop_columns(metas, datas, data_id)

    st.header("7. Manual Handling")
    subtabs7 = st.tabs(tab_subtitles)
    with subtabs7[0]:
        data_id = data_ids[0]
        meta, data = metas[data_id], datas[data_id]
        show_manual_handling(metas, datas, data_id)
    with subtabs7[1]:
        data_id = data_ids[1]
        show_manual_handling(metas, datas, data_id)

    st.header("8. Except Diabetes-relevant Columns")
    subtabs8 = st.tabs(tab_subtitles)
    with subtabs8[0]:
        data_id = data_ids[0]
        meta, data = metas[data_id], datas[data_id]
        show_drop_diabetes_columns(metas, datas, data_id)
        datas[data_id].to_feather(PATH.family_final_data)
    with subtabs8[1]:
        data_id = data_ids[1]
        show_drop_diabetes_columns(metas, datas, data_id)
        datas[data_id].to_feather(PATH.sample_adult_final_data)

    st.header("9. Correlation Coefficients")
    show_correlations()


with tabs[3]:
    st.header("1. Data Selection")
    st.write("#### 1.1 Merge Family and Sample Adult Dataset")
    data, meta = select_dataset()
    st.dataframe(data)

    st.write("#### 1.2 Split Dataset")
    dataset = split_dataset(data)

    st.header("2. Logistic Regression (10 Folds Soft Voting Ensemble)")
    show_logstic_regression(data, meta, dataset)

    st.header("3. Random Forest Classifier (10 Folds Soft Voting Ensemble)")
    show_random_forest_classifier(meta, dataset)
