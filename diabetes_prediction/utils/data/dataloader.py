"""Dataloader module
"""
from diabetes_prediction._utils import *

from tabula import read_pdf


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
    data = pd.read_csv(output_path)

    # Evaluate columns (dictionary or list)
    for col in ['options', 'keywords']:
        data[col] = data[col].map(eval)

    return data


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
