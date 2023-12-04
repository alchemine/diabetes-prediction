"""Tab page for streamlit app
"""

from diabetes_prediction.utils.app.io import *
from diabetes_prediction.utils.data.dataloader import *
from diabetes_prediction.utils.data.preprocessing import *

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


get_score = lambda y, p: dict(f1_score=f1_score(y, p), accuracy=accuracy_score(y, p))
select_num_cols = lambda corr, threshold: corr[corr['importance'].cumsum() <= threshold].index if threshold < 1 else corr.index[:threshold]
select_cat_cols = lambda fis, threshold: fis[fis.cumsum() <= threshold].dropna().index if threshold < 1 else fis.index[:threshold]

def training(base_estimator, X_tv, y_tv, n_splits=10, verbose=False):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=PARAMS.seed)  # default: cv=5

    models = []
    scores = []
    for idxs_train, idxs_test in cv.split(X_tv, y_tv):
        clone_model = clone(base_estimator)
        X_train_fold, y_train_fold = X_tv.loc[idxs_train], y_tv.loc[idxs_train]
        X_val_fold, y_val_fold = X_tv.loc[idxs_test], y_tv.loc[idxs_test]

        clone_model.fit(X_train_fold, y_train_fold)
        p_train_fold = clone_model.predict(X_train_fold)
        p_val_fold = clone_model.predict(X_val_fold)

        train_score, val_score = get_score(y_train_fold, p_train_fold), get_score(y_val_fold, p_val_fold)
        score = {'train_acc': train_score['accuracy'], 'train_f1': train_score['f1_score'],
                 'val_acc': val_score['accuracy'], 'val_f1': val_score['f1_score']}
        if verbose:
            print(score)

        models.append(clone_model)
        scores.append(score)

    scores = pd.DataFrame(scores).T
    scores = pd.concat([scores, scores.mean(axis=1)], axis=1)
    scores.columns = [f'fold_{i}' for i in range(10)] + ['mean']
    return models, scores


def test(models, X_test, y_test):
    proba_test = np.array([model.predict_proba(X_test) for model in models]).mean(axis=0)
    p_test = proba_test.argmax(axis=1)
    scores = get_score(y_test, p_test)
    print("Test score:", scores)
    return scores


def select_dataset():
    metas, datas = load_dataset()
    family       = pd.read_feather(PATH.family_final_data)
    sample_adult = pd.read_feather(PATH.sample_adult_final_data)

    data = pd.merge(sample_adult, family, how='left', on='family_id')
    data = data[data['label'][0, 1])]
    meta = pd.concat([metas['family'], metas['sample_adult']])
    return data, meta

def split_dataset(data):
    train_val_data, test_data = train_test_split(data, test_size=0.4, stratify=data['label'], random_state=PARAMS.seed)
    train_val_data.reset_index(drop=True, inplace=True)

    X_tv = train_val_data.drop(columns='label')
    y_tv = train_val_data['label']

    X_test = test_data.drop(columns='label')
    y_test = test_data['label']

    with st_stdout("code"):
        print(f"Train + validation data: {len(train_val_data)}")
        print(f"Test data: {len(test_data)}")

    return dict(
        X_tv=X_tv, y_tv=y_tv,
        X_test=X_test, y_test=y_test
    )

def show_logstic_regression(data, meta, dataset):
    n_cols  = 30
    corr    = get_corr(data)
    df_desc = get_description(meta, corr.index[:n_cols])
    fig, ax = plt.subplots(figsize=(30, 3))
    corr['corr_abs'][:n_cols].plot.bar(title=f"abs(Correlation Coefficient) (top {n_cols} features from {len(corr)} features)", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

    st.write("#### Feature Selection using Cumulative Sum of Correlation Coefficients")
    for threshold in [0.9, 0.6, 0.3, 0.2, 0.1, 0.05]:
        cols = select_num_cols(corr, threshold)
        model = LogisticRegression(random_state=PARAMS.seed)
        models, train_val_scores = training(model, dataset['X_tv'][cols], dataset['y_tv'])
        test_scores = test(models, dataset['X_test'][cols], dataset['y_test'])

        with st_stdout("code"):
            print(f"Threshold: {threshold} (# columns: {len(cols)})")
            print(f"Test score: {test_scores}")
        st.dataframe(train_val_scores)


def show_random_forest_classifier(meta, dataset):
    model = RandomForestClassifier(random_state=PARAMS.seed, n_jobs=-1)
    models, train_val_scores = training(model, dataset['X_tv'], dataset['y_tv'])
    fis = pd.DataFrame(np.mean([model.feature_importances_ for model in models], axis=0),
                       index=models[0].feature_names_in_, columns=['feature_importance'])
    fis.sort_values('feature_importance', ascending=False, inplace=True)

    n_cols = 30
    df_desc = get_description(meta, fis.index[:n_cols])
    fig, ax = plt.subplots(figsize=(30, 3))
    fis.head(n_cols).plot.bar(legend=False, title=f"Feature importances from RandomForestClassifier (average of 10 models, {n_cols} features from {len(fis)} features)", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(df_desc)

    st.write("#### Feature Selection using Cumulative Sum of Mean Decrease in Impurity")
    for threshold in [0.9, 0.6, 0.3, 0.2, 0.1, 0.05]:
        cols  = select_cat_cols(fis, threshold)
        model = RandomForestClassifier(random_state=PARAMS.seed, n_jobs=-1)
        models, train_val_scores = training(model, dataset['X_tv'][cols], dataset['y_tv'])
        test_scores = test(models, dataset['X_test'][cols], dataset['y_test'])

        with st_stdout("code"):
            print(f"Threshold: {threshold} (# columns: {len(cols)})")
            print(f"Test score: {test_scores}")
        st.dataframe(train_val_scores)
