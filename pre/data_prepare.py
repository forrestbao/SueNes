import tensorflow_datasets as tfds 


def load_pairs(dataset_name, split, take_percent, features ):
    """Load pairs of documents and their summaries

    dataset_name: str, a unique name to ref to it in tfds 
        e.g., 'cnn_dailymail', 'newsroom', 'big_patent'

    split: str, 'test', 'train' or 'validation' 
           results in a _OptionsDataset type 

    take_percent: int, 0 to 100, ratio of data to take from the dataset or data split. 100 means use all data 

    features: [str, str], names of the document and the summary in TFDS, e.g., ['article', 'highlights']

    """

    print ("Loading data. If the data not available locally, data will be downloaded first.")

    dataset = tfds.load(name=dataset_name, split=
            split+ '[{}%:{}%]'.format(0, take_percent)
            )

    pairs = [(piece[features[0]], piece[features[1]])  for piece in dataset]

    return pairs 

def cross_pair(data_pairs):
    pass

def mutate(data_pairs):
    pass

if __name__ == "__main__":
    import data_conf
    pairs = load_pairs(data_conf.dataset_name, data_conf.split, data_conf.take_percent, data_conf.features)
    


