"""Modeling module.
"""

from diabetes_prediction._utils import *

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

get_score = lambda y, p: dict(accuracy=accuracy_score(y, p), f1_score=f1_score(y, p, pos_label=0))


def experiment(metadata, dataset_proc, base_estimator, feature_selector, plot=True):
    split_Xy = lambda data, features: (data[features], data[PARAMS.target])

    rst = {}
    for thr in tqdm([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
        features = feature_selector.select(thr)

        X_train, y_train = split_Xy(dataset_proc['train'], features)
        X_val,   y_val   = split_Xy(dataset_proc['val'],   features)
        X_test,  y_test  = split_Xy(dataset_proc['test'],  features)

        models, scores = train(base_estimator, X_train, y_train, n_folds=10)
        # models, scores = train(base_estimator, X_train, y_train, n_folds=3)
        val_scores  = test(models, X_val, y_val)
        test_scores = test(models, X_test, y_test)

        scores['validation'] = [None, None, *val_scores.values()]
        scores['test'] = [None, None, *test_scores.values()]
        scores.index.name = f"Threshold: {thr} (# features: {len(features)})"
        rst[thr] = {'scores': scores, 'features': features, 'metadata': merge_features_metadata(features, metadata)}

    if plot:
        summary = {}
        for metric in ('acc', 'f1'):
            summary[metric] = pd.DataFrame()
            for thr in rst:
                s = rst[thr]['scores']
                cv_train_scores = s.loc[[f'train_{metric}'], 'fold_mean']
                cv_test_scores  = s.loc[[f'test_{metric}'], 'fold_mean']
                val_scores      = s.loc[[f'test_{metric}'], 'validation']
                test_scores     = s.loc[[f'test_{metric}'], 'test']
                cur_summary     = pd.DataFrame(np.c_[cv_train_scores, cv_test_scores, val_scores, test_scores], index=[thr], columns=['train(cv)', 'test(cv)', 'validation', 'test'])
                summary[metric] = pd.concat([summary[metric], cur_summary])

        # select optimal threshold based on validation score
        opt_thr = summary['f1']['validation'].idxmax()

        fig, axes = plt.subplots(2, figsize=PARAMS.figsize)
        for ax, metric in zip(axes.flat, ('f1', 'acc')):
            summary[metric].drop(columns='test').plot.bar(ax=ax, rot=0, ylabel=metric)
            ax.axhline(summary[metric].loc[opt_thr, 'test'], color='k', linestyle='--', label=f'test (optimal threshold: {opt_thr})')
            if ax is axes[0]:
                ax.legend().remove()
                ax.set_xticklabels([])
            else:
                ax.legend(loc='lower left')
                ax.set_xlabel('Threshold')

    return rst, summary


def train(base_estimator, X_train, y_train, n_folds=10, verbose=False):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=PARAMS.seed)  # default: cv=5
    models, scores = [], []
    for idxs_train, idxs_val in cv.split(X_train, y_train):
        clone_model = clone(base_estimator)

        X_train_fold, y_train_fold = X_train.loc[idxs_train], y_train.loc[idxs_train]
        X_val_fold, y_val_fold = X_train.loc[idxs_val], y_train.loc[idxs_val]

        clone_model.fit(X_train_fold, y_train_fold)
        p_train_fold = clone_model.predict(X_train_fold)
        p_val_fold = clone_model.predict(X_val_fold)

        train_score, val_score = get_score(y_train_fold, p_train_fold), get_score(y_val_fold, p_val_fold)
        score = {'train_acc': train_score['accuracy'], 'train_f1': train_score['f1_score'],
                 'test_acc': val_score['accuracy'], 'test_f1': val_score['f1_score']}
        if verbose:
            print(score)

        models.append(clone_model)
        scores.append(score)

    scores = pd.DataFrame(scores, index=[f'fold_{1 + i}' for i in range(n_folds)]).T
    scores['fold_mean'] = scores.mean(axis=1)
    scores['fold_std'] = scores.std(axis=1)
    return models, scores


def test(models, X, y):
    proba = np.array([model.predict_proba(X) for model in models]).mean(axis=0)
    p = proba.argmax(axis=1)
    return get_score(y, p)
