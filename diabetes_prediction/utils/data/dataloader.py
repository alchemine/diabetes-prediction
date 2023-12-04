"""Dataloader module
"""

from diabetes_prediction._utils import *
from diabetes_prediction.utils.data.preprocessing import *

from tabula import read_pdf
from sklearn.model_selection import train_test_split


@T
def load_dataset(data_id: str, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset for `data_id`.
    Dataset consists of metadata and data(records).

    Args:
        data_id: Data ID.
            Should be ('family', 'household', 'person', 'sample_child', 'sample_adult').
        overwrite: Whether to overwrite metadata or not.

    Returns:
        Metadata and data.
    """
    paths    = PATH.get(data_id)
    metadata = load_metadata(paths, overwrite)
    data     = pd.read_csv(paths['data'], dtype=str)
    return metadata, data


@T
def load_metadata(paths: dict, overwrite: bool) -> pd.DataFrame:
    """Load metadata for `data_id`.

    Args:
        paths: Dictionary of paths.
        overwrite: Whether to overwrite metadata or not.

    Returns:
        Metadata having summary and layout.
    """
    # Generate summary and layout
    output_path = paths['metadata']
    if overwrite or not exists(output_path):
        summary  = _get_summary(paths['summary'])
        layout   = _get_layout(paths['layout'], summary)
        metadata = _merge_summary_layout(summary, layout)
        metadata.to_csv(output_path, index=False)
    return read_metadata(output_path)


@T
def _get_summary(path):
    def _load_data(path):
        columns = [93, 126, 150, 200, 320, 390, 660, 703]
        return read_pdf(path, columns=columns, guess=False, pages='all', silent=True)
    def _process(raw_datas):
        # 1. Drop head contents and concatenate
        data = pd.concat(
            [pd.DataFrame(data.loc[5:]).rename(columns=data.loc[4]) for data in raw_datas],
            ignore_index=True
        )

        # 2. Handle the appended rows
        # Find the appended row
        idxs_app = data[data['Question #'].isna() & data['FinalDocName'].isna()].index

        # Concatenate values
        for idx in reversed(idxs_app):
            dict_app = data.loc[idx].dropna().to_dict()
            for key, value in dict_app.items():
                data.loc[idx - 1, key] += f" {value}"

        # Remove appended rows
        data = data.drop(idxs_app).reset_index(drop=True)

        # 3. Remove space in location
        data['Location'] = data['Location'].str.replace(' ', '')

        # 4. Drop the last row
        data = data[:-1]

        _sanity_check(data)
        return data
    def _sanity_check(data):
        try:
            # Columns below should not contain space
            for col in ['FinalDocName']:
                assert len(data[data[col].str.contains(' ')]) == 0

            # Columns below should start with number
            for col in ('Location', 'Length'):
                assert len(data[~data[col].str.contains("^[0-9]", regex=True)]) == 0
        except Exception as e:
            display(data)
            print(e)

    raw_datas = _load_data(path)
    data      = _process(raw_datas)
    return data


@T
def _get_layout(path, summary):
    def _get_options(path, summary):
        def _load_data(path):
            columns = [115]
            return read_pdf(path, columns=columns, guess=False, pages='all', silent=True)
        def _process(raw_data, summary):
            # 1. Preparation
            raw_data.columns = ['key', 'value']
            idxs_sep = raw_data[raw_data.iloc[:, 0].fillna('').str.contains('Question ID')].index

            # 2. Fill options data
            df_options = pd.DataFrame(columns=['FinalDocName', 'Options'])
            for it in range(len(idxs_sep)):
                # Select current question id(or final documentation name)
                if it < len(idxs_sep) - 1:
                    cur_data = raw_data[idxs_sep[it]:idxs_sep[it + 1]]
                else:
                    cur_data = raw_data[idxs_sep[it]:]
                final_doc_name = raw_data.loc[idxs_sep[it] + 1]['value'].split(': ')[1]

                options = {}
                for idx in reversed(cur_data.index):
                    # Select valid options checked by the length
                    row = cur_data.loc[idx]
                    length = int(summary[summary['FinalDocName'] == final_doc_name]['Length'].item())
                    if re.search(r"^[0-9]{0,%d}([-]*[0-9]{%d})*$" % (length, length), str(row['key'])):  # row['key'] can be nan(float)
                        options[row['key']] = row['value']
                    else:
                        break
                df_options = pd.concat([
                    df_options,
                    pd.DataFrame({'FinalDocName': [final_doc_name], 'Options': json.dumps(options)})
                ], ignore_index=True)

            return df_options

        raw_datas = _load_data(path)
        return pd.concat([_process(raw_data, summary) for raw_data in raw_datas], ignore_index=True)
    def _get_keywords(path):
        def _load_data(path):
            columns = [97]
            return read_pdf(path, columns=columns, guess=False, pages='all', silent=True)

        def _process(raw_data):
            # 1. Preparation
            raw_data.columns = ['key', 'value']
            idxs_sep = raw_data[raw_data.iloc[:, 0].fillna('').str.contains('Question ID')].index

            # 2. Fill keywords data
            df_keywords = pd.DataFrame(columns=['FinalDocName', 'Keywords'])
            for it in range(len(idxs_sep)):
                # Select current question id(or final documentation name)
                if it < len(idxs_sep) - 1:
                    table = raw_data[idxs_sep[it]:idxs_sep[it + 1]]
                else:
                    table = raw_data[idxs_sep[it]:]
                final_documentation_name = raw_data.loc[idxs_sep[it] + 1]['value'].split(': ')[1]

                # Select keywords
                value = table[table['key'] == 'Keywords:']['value'].item()
                value = [s.lower() for s in value.split('; ')] if value != 'None' else []

                df_keywords = pd.concat([
                    df_keywords,
                    pd.DataFrame({'FinalDocName': [final_documentation_name], 'Keywords': json.dumps(value)})
                ], ignore_index=True)
            return df_keywords

        raw_datas = _load_data(path)
        return pd.concat(lmap(_process, raw_datas), ignore_index=True)

    options  = _get_options(path, summary)
    keywords = _get_keywords(path)
    layout   = options.join(keywords, rsuffix='_').drop(columns='FinalDocName_')
    return layout


@T
def _merge_summary_layout(summary, layout):
    # 1. Join summary data and layout data
    data = summary.join(layout, rsuffix='_').drop(columns='FinalDocName_')

    # 2. Change nan to None
    data.replace(np.nan, None, inplace=True)

    # 3. Change dtypes to string
    for col in data:
        data[col] = data[col].astype(str)

    # 4. Strip, lower all columns
    for col in data:
        data[col] = data[col].str.strip()
        if col not in ('Question #', 'FinalDocName'):
            data[col] = data[col].str.lower()

    # 5. Change none to None
    data.replace('none', None, inplace=True)

    # 6. Rename columns
    data.rename(columns={
        'Question #': 'question_id',
        'Recode': 'recode',
        'Instrument Variable Name': 'inst_id',
        'FinalDocName': 'final_id',
        'Processing Variable Label': 'description',
        'Location': 'location',
        'Length': 'value_length',
        'Options': 'options',
        'Keywords': 'keywords'
    }, inplace=True)

    # 7. Select columns
    cols = ['question_id', 'final_id', 'description', 'options', 'keywords']
    return data[cols]


@T
def load_merged_datas(metadatas:dict, datas: dict, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge multiple DataFrames into one DataFrame using `family_id`.
    See details in diabetes_prediction.utils.data.preprocessing.extract_family_id().

    Args:
        metadatas: Dictionary of metadata.
        datas: Dictionary of data.
        overwrite: Whether to overwrite metadata or not.

    Returns:
        Merged metadata and data DataFrame.
    """
    paths = PATH.get_proc()
    if not exists(paths['data']) or overwrite:
        family       = copy(datas['family'])
        sample_adult = copy(datas['sample_adult'])

        # 1. Extract family_id
        extract_family_id(family)
        extract_family_id(sample_adult)

        # 2. Remove redundant columns
        cols = ['FMX', 'HHX', 'SRVY_YR', 'RECTYPE']
        family.drop(columns=cols, inplace=True)
        sample_adult.drop(columns=cols, inplace=True)

        # 3. Join datas
        metadata = pd.concat([metadatas['family'], metadatas['sample_adult']])
        metadata = metadata.drop_duplicates(subset=['final_id'])
        data     = pd.merge(sample_adult, family, how='left', on='family_id').drop(columns='family_id')

        # 4. Save
        os.makedirs(PATH.proc, exist_ok=True)
        metadata.to_csv(paths['metadata'], index=False)
        data.to_csv(paths['data'], index=False)

    # 5. Load
    metadata = read_metadata(paths['metadata'])
    data     = pd.read_csv(paths['data'])
    return metadata, data


@T
def split_data(data_: pd.DataFrame, drop_unknown: bool, test_size: float = 0.3) -> dict:
    """Split data into dataset with train, validation, test set.

    Args:
        data_: Data records.
        drop_unknown: Whether to drop records with unknown label or not.
        test_size: Proportion of test set.

    Returns:
        Dataset with train, validation, test set.
    """
    def _drop_unknown_label_rows(data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Drop records with unknown label.

        Args:
            data: Data records.
            target: Target column name.

        Returns:
            Data records without unknown label.
        """
        data[target] = data[target].astype(int)
        unknown_idxs = data[~data[target].isin([1, 2])].index
        print(f"Remove records with unknown label ({len(unknown_idxs)} records from {len(data)} records).")
        data.drop(unknown_idxs, inplace=True)


    if not drop_unknown:
        raise ValueError(f"Invalid drop_unknown: {drop_unknown}. drop_unknown should be True.")

    data   = copy(data_)
    target = PARAMS.target

    # 1. Remove records with unknown label
    _drop_unknown_label_rows(data, target)

    # 2. Split data
    train_val_data, test_data = train_test_split(data, test_size=test_size, stratify=data[target], random_state=PARAMS.seed)
    train_data, val_data = train_test_split(train_val_data, test_size=test_size, stratify=train_val_data[target], random_state=PARAMS.seed)

    # 3. Clean index
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    return dict(train=train_data, val=val_data, test=test_data)


@T
def load_processed_dataset(metadata: pd.DataFrame = None, dataset: dict = None, overwrite: bool = False) -> dict:
    """Preprocess and load processed dataset.

    Args:
        metadata: Metadata.
        dataset: Dataset with train, validation, test set.
        overwrite: Whether to overwrite processed dataset or not.

    Returns:
        Preprocessed dataset.
    """
    paths = PATH.get_proc()
    if not exists(paths['test']) or overwrite:
        pp = Preprocessor(metadata)
        dataset_proc = {}
        dataset_proc['train'] = pp.fit_transform(dataset['train'])
        dataset_proc['val']   = pp.transform(dataset['val'])
        dataset_proc['test']  = pp.transform(dataset['test'])
        for key, data in dataset_proc.items():
            data.to_feather(paths[key])
    return {key: pd.read_feather(paths[key]) for key in ('train', 'val', 'test')}
