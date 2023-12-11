from diabetes_prediction.utils.data import *


if __name__ == '__main__':
    data_ids = ['family', 'sample_adult', 'sample_child']

    metadatas, datas = {}, {}
    for data_id in data_ids:
        metadatas[data_id], datas[data_id] = load_dataset(data_id)

    metadata, data = load_merged_datas(metadatas, datas)
    dataset = split_dataset(data, drop_unknown=True)
    dataset_proc = load_processed_dataset(metadata, dataset, overwrite=True)