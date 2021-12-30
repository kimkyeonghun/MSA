import pickle

import numpy as np

DATA_PATH = './sdk_features'

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def parse_UR_FUNNY():
    data_folds = load_pickle(DATA_PATH + '/data_folds.pkl')
    train_split = data_folds['train']
    dev_split = data_folds['dev']
    test_split = data_folds['test']

    word_aligned_openface_sdk = load_pickle(DATA_PATH + "/openface_features_sdk.pkl")
    word_aligned_covarep_sdk = load_pickle(DATA_PATH + "/covarep_features_sdk.pkl")
    word_embedding_idx_sdk=load_pickle(DATA_PATH + "/language_sdk.pkl")
    humor_label_sdk = load_pickle(DATA_PATH + '/humor_label_sdk.pkl')

    EPS = 1e-6

    train = []
    dev = []
    test = []

    num_drop = 0

    for key in humor_label_sdk.keys():
        label = np.array(humor_label_sdk[key], dtype = int)
        _word_id = np.array(word_embedding_idx_sdk[key]['punchline_features'])
        _acoustic = np.array(word_aligned_covarep_sdk[key]['punchline_features'])
        _visual = np.array(word_aligned_openface_sdk[key]['punchline_features'])
        if not _word_id.shape[0] == _acoustic.shape[0] == _visual.shape[0]:
            num_drop += 1
            continue

        label = np.array([np.nan_to_num(label)])[:, np.newaxis]
        _visual = np.nan_to_num(_visual)
        _acoustic = np.nan_to_num(_acoustic)

        actual_words = []
        visual = []
        acoustic = []
        for i, word in enumerate(_word_id):
            actual_words.append(word)
            visual.append(_visual[i, :])
            acoustic.append(_acoustic[i, :])

        words = np.asarray(actual_words)
        visual = np.asarray(visual)
        acoustic = np.asarray(acoustic)

        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
        acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

        if key in train_split:
            train.append(((words, visual, acoustic), label, key))
        elif key in dev_split:
            dev.append(((words, visual, acoustic), label, key))
        elif key in test_split:
            test.append(((words, visual, acoustic), label, key))
        else:
            print(f"Found video that doesn't belong to any splits: {key}")

    print(f"# of Train {len(train)}")
    print(f"# of dev {len(dev)}")
    print(f"# of test {len(test)}")
    print(f"Total number of {num_drop} datapoints have been dropped.")

    return train, dev, test

def save(train, val, test, datasetName):
    allDataset = {"train":train, "val" : val, "test": test}
    with open('cmu_{}.pkl'.format(datasetName),'wb') as f:
        pickle.dump(allDataset,f)
    print("Save Complete!")

if __name__ == '__main__':
    train, dev, test = parse_UR_FUNNY()
    save(train, dev, test, 'ur_funny')
