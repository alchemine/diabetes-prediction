"""Data preprocessing module
"""

from diabetes_prediction._utils import *
from sklearn.impute import SimpleImputer


def load_dataset():
    # 1. Read raw dataset
    metas = dict(
        family=pd.read_csv(PATH.family_meta),
        sample_child=pd.read_csv(PATH.sample_child_meta),
        sample_adult=pd.read_csv(PATH.sample_adult_meta)
    )

    datas = dict(
        family=pd.read_csv(PATH.family_data, dtype=str),
        sample_child=pd.read_csv(PATH.sample_child_data, dtype=str),
        sample_adult=pd.read_csv(PATH.sample_adult_data, dtype=str)
    )

    # 2. Preprocess meta data
    for data_id in metas:
        preprocess_meta(metas[data_id])

    return metas, datas


def split_data(data_, target):
    return data_

def preprocess_data(data_, meta, target=None):
    data = data_.copy()

    # Replace options and impute
    replace_ambiguous_options(data, meta)

    # Impute
    impute_data(data, meta)

    # Change dtypes
    set_dtypes(data, meta)

    # Impute numerical features
    impute_numerical_features(data, meta)

    # Extract features
    extract_features(data, target)

    # Drop constant / redundant / imbalance columns
    drop_columns(data)

    # Manual handling
    manual_handling(data, meta)

    # Except diabetes-relevant columns
    drop_diabetes_columns(data, meta)

    return data

@T
def replace_ambiguous_options(data, meta):
    # Replace ambiguous options into 'unknown`
    #  don't know
    #  not ascertained
    #  refused
    #  unknown
    replacement_map = {src: "unknown" for src in ("don't know", "don’t know", "not ascertained", "refused", "not available", "time period format", "undefined", "undefinable")}

    for col in data:
        # 1. Replace options with words ('1' -> 'yes')
        options   = get_meta_values(meta, col)['options']  # {'1': 'yes', '2': 'no', '3': 'don\'t know', '4': 'not ascertained', '5': 'refused', '-1': 'unknown'}
        data[col] = data[col].replace(options)

        # 2. Replace ambiguous options with 'unknown'
        data[col] = data[col].replace(replacement_map)

        # 3. Replace words with options (restore)
        options_inversed = inverse_dict(options)           # {'yes': '1', 'no': '2', 'don\'t know': '3', 'not ascertained': '4', 'refused': '5', 'unknown': '-1'}
        data[col] = data[col].replace(options_inversed)

@T
def impute_data(data, meta):
    # Impute nan values with 'unknown' value
    for col in data:
        options_inversed = inverse_dict(get_meta_values(meta, col)['options'])
        imputer = SimpleImputer(strategy='constant', fill_value=options_inversed['unknown'])
        data[[col]] = imputer.fit_transform(data[[col]])

@T
def set_dtypes(data, meta):
    # Numerical features: only have '-1'('unknown') or '-' in option or YEAR
    num_features = []
    for col in data:
        options = get_meta_values(meta, col)['options']
        if list(options) == ['-1']:
            num_features.append(col)
        else:
            for key in options:
                if re.search("^[0-9]+-[0-9]+$", key):
                    num_features.append(col)
                    break

    # Manual filtering
    # Append features
    num_features_app  = [col for col in ('ASISAD', 'ASIMUCH') if col in data]
    for words in ("how satisfied", "total number", "get sick or have accident", "days 5+/4+ drinks", "alcohol drinking", "strength activity", "freq ", "work status", "confidence", "time since", "received calls", "cost of", "education of", "total combined", "ratio of", "how difficult", "duration of", "agree/disagree", "length", "how worried", "how often", "how long", "time ago", "most recent", "year of", "time period", "degree of difficulty", "diff ", "frequency"):
        num_features_app += [col for col in data if get_meta_values(meta, col)['description'].startswith(words)]
    for final_id in ("RPAP", "HEAR_", "COG_"):
        num_features_app += [col for col in data if get_meta_values(meta, col)['final_id'].startswith(final_id)]

    num_features = list(set(num_features + num_features_app))

    # Except features
    num_features = [col for col in num_features if col not in ('HHX', 'FMX', 'FPX', 'SRVY_YR', 'OCCUPN1', 'OCCUPN2', 'INDSTRN1', 'INDSTRN2')]

    # Categorical features: otherwise
    cat_features = data.columns.drop(num_features)

    # Apply
    for col in num_features:  # error at once
        data[col] = data[col].astype('f')
    data[cat_features] = data[cat_features].astype(str)

@T
def impute_numerical_features(data, meta):
    # Fill numeric unknown values with mode (zero-centered or long tailed distribution)
    for col in data.select_dtypes('number').columns:
        data[col] = data[col].replace(float(get_unknown_value(meta, col)), None)
        data[col] = data[col].fillna(data[col].mode()[0])  # fill with mode

@T
def extract_features(data, target):
    # Add family indicator (family_id)
    extract_family_id(data)

    # Extract label
    if target:
        extract_label(data, target)

@T
def extract_family_id(data):
    # - FMX: Use this variable in combination with HHX and SRVY_YR(constant, 2018) to identify individual families.
    data['family_id'] = data['HHX'] + data['FMX'] + data['SRVY_YR']

@T
def extract_label(data, target):
    data.loc[data[target] == '1', 'label'] = 1
    data.loc[data[target] == '2', 'label'] = 0
    data.loc[~data[target].isin(['1', '2']), 'label'] = 2

@T
def drop_columns(data):
    # Drop constant columns
    cnts = data.nunique().sort_values()
    constant_cols = cnts[cnts == 1].index
    data.drop(columns=constant_cols, inplace=True)

    # Drop redundant columns
    redundant_cols = ['HHX', 'FMX', 'INTV_QRT']
    cols = [col for col in redundant_cols if col in data]
    data.drop(columns=cols, inplace=True)

    # *CAUTION* Drop imbalanced distributed columns
    # imbalance_cols = []
    # for col in data:
    #     if col == 'label':  # except label
    #         continue
    #     max_cnt = data[col].value_counts(normalize=True).values[0]
    #     if max_cnt >= 0.85:
    #         imbalance_cols.append(col)
    # data.drop(columns=imbalance_cols, inplace=True)

@T
def manual_handling_sample_child(data, meta):
    # 1. Numerical feature + unknown
    for col in ('BWTGRM_P', 'TOTOZ_P', 'CHGHT_TC', 'CWGHT_TC', 'BMI_SC', 'MHIBOY2', 'MHIGRL2'):
        data[col] = data[col].replace(get_unknown_value(meta, col), None).astype('f')
        data[col] = data[col].fillna(data[col].mode()[0])  # fill with mode (zero-centered or long tailed distribution)

    # 2. Special handling
    col = 'SCHDAYRP'
    data[col] = data[col].astype('f').replace(996, 41)    # Did not go to school -> missed over 40

    col = 'CWZMSWKP'
    data[col] = data[col].astype('f').replace(995, int(get_unknown_value(meta, col)))  # Home schooled -> unknown
    data[col] = data[col].astype('f').replace(996, 11)    # Did not go to daycare, preschool, school or work -> missed over 10

@T
def manual_handling(data, meta):
    # Handle 'do nothing' options
    for col in data:
        if col in ('label', 'family_id'):
            continue
        options = get_meta_values(meta, col)['options']
        options_inversed = inverse_dict(options)
        for option in options_inversed:
            if re.search("^unable to do ", option) or (option == 'never'):
                value = float(options_inversed[option])
                data[col] = data[col].replace(value, 0)  # never do that

@T
def drop_diabetes_columns(data, meta):
    # Drop columns which have diabetes keywords
    idxs = meta['keywords'].astype(str).str.contains('diabetes')
    diabetes_cols = meta.loc[idxs[idxs].index, 'final_id']
    diabetes_cols = [col for col in diabetes_cols if col in data]
    data.drop(columns=diabetes_cols, inplace=True)
