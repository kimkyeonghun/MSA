import numpy as np
import pickle
from collections import defaultdict
from transformers import BertTokenizer

max_l = 100

def save(train, val, test, dataset_name):
    allDataset = {"train": train, "val": val, "test": test}
    with open('{}.pkl'.format(dataset_name),'wb') as f:
        pickle.dump(allDataset,f)
    print("Save Complete!")

def get_word_indices(tokenizer, data_x):
    length = len(data_x)
    input_ids = tokenizer.convert_tokens_to_ids(data_x)
    return np.array(input_ids + [0]*(max_l-length))[:max_l]

def get_dialogue_ids(keys):
    ids=defaultdict(list)
    for key in keys:
        ids[key.split("_")[0]].append(int(key.split("_")[1]))
    for ID, utts in ids.items():
        ids[ID]=[str(utt) for utt in sorted(utts)]
    return ids

def get_max_utts(train_ids, val_ids, test_ids):
    max_utts_train = max([len(train_ids[vid]) for vid in train_ids.keys()])
    max_utts_val = max([len(val_ids[vid]) for vid in val_ids.keys()])
    max_utts_test = max([len(test_ids[vid]) for vid in test_ids.keys()])
    return np.max([max_utts_train, max_utts_val, max_utts_test])

def get_dialogue_text(dialogue_ids, train, val, test):
    key = list(train.keys())[0]
    pad = [0]*len(train[key][0])
    print(len(pad))
    def get_emb(dialogue_id, local_data):
        dialogue_text = []
        for vid in dialogue_id.keys():
            local_text = []
            for utt in dialogue_id[vid]:
                local_text.append(local_data[vid+"_"+str(utt)][0][:])
            for _ in range(33-len(local_text)):
                local_text.append(pad[:])
            dialogue_text.append(local_text[:33])
        return np.array(dialogue_text)
    train_dialogue_features = get_emb(dialogue_ids[0], train)
    val_dialogue_features = get_emb(dialogue_ids[1], val)
    test_dialogue_features = get_emb(dialogue_ids[2], test)
    return train_dialogue_features, val_dialogue_features, test_dialogue_features
    
def get_one_hot(label):
    label_arr = [0]*3
    label_arr[label]=1
    return label_arr[:]

def parse_MELD():
    revs, _, word_idx_map, _, _, label_index = pickle.load(open("./MELD/data_sentiment.p", "rb"))
    # train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open("./MELD/audio_embeddings_feature_selection_sentiment.pkl", "rb"))
    # train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open("./MELD/audio_sentiment.pkl", 'rb'))
    
    print("Labels used for this classification: ", label_index)

    train_data, val_data, test_data = {}, {}, {}
    for i in range(len(revs)):
        utterance_id = revs[i]['dialog']+"_"+revs[i]['utterance']
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        sentence_word_indices = get_word_indices(tokenizer, tokenizer.tokenize(revs[i]['text']))
        #sentence_word_indices = revs[i]['text']
        label = label_index[revs[i]['y']]

        if revs[i]['split']=="train":
            train_data[utterance_id]=(sentence_word_indices,label)
        elif revs[i]['split']=="val":
            val_data[utterance_id]=(sentence_word_indices,label)
        elif revs[i]['split']=="test":
            test_data[utterance_id]=(sentence_word_indices,label)

    train_dialogue_ids = get_dialogue_ids(train_data.keys())
    val_dialogue_ids = get_dialogue_ids(val_data.keys())
    test_dialogue_ids = get_dialogue_ids(test_data.keys())

    max_utts = get_max_utts(train_dialogue_ids, val_dialogue_ids, test_dialogue_ids)
    # print(train_dialogue_ids)
    # print(val_dialogue_ids)
    # print(test_dialogue_ids)

    dialogue_ids = (train_dialogue_ids, val_dialogue_ids, test_dialogue_ids)
    train_text_x, val_text_x, test_text_x = get_dialogue_text(dialogue_ids, train_data, val_data, test_data)
    train_audio_x, val_audio_x, test_audio_x = pickle.load(open("./MELD/audio_sentiment.pkl", 'rb'), encoding='latin1')
    def concatenate_fusion(ID, text, audio):
        bimodal=[]
        for i,(vid, utts) in enumerate(ID.items()):
            bimodal.append((text[i],audio[vid]))
        return bimodal

    train_dialogue_features = concatenate_fusion(train_dialogue_ids, train_text_x, train_audio_x)
    val_dialogue_features = concatenate_fusion(val_dialogue_ids, val_text_x, val_audio_x)
    test_dialogue_features = concatenate_fusion(test_dialogue_ids, test_text_x, test_audio_x)

    train_dialogue_length, val_dialogue_length, test_dialogue_length = [], [], []
    for vid, utts in train_dialogue_ids.items():
        train_dialogue_length.append(len(utts))
    for vid, utts in val_dialogue_ids.items():
        val_dialogue_length.append(len(utts))
    for vid, utts in test_dialogue_ids.items():
        test_dialogue_length.append(len(utts))

    def get_labels(ids, data):
        dialogue_label=[]

        for vid, utts in ids.items():
            local_labels=[]
            for utt in utts:
                local_labels.append(get_one_hot(data[vid+"_"+str(utt)][1]))
            for _ in range(33-len(local_labels)):
                local_labels.append(get_one_hot(1)) # Dummy label
            dialogue_label.append(local_labels[:max_utts])
        return np.array(dialogue_label)

    train_dialogue_label=get_labels(train_dialogue_ids, train_data)
    val_dialogue_label=get_labels(val_dialogue_ids, val_data)
    test_dialogue_label=get_labels(test_dialogue_ids, test_data)

    train_mask = np.zeros((len(train_dialogue_length), max_utts), dtype='float')
    for i in range(len(train_dialogue_length)):
        train_mask[i,:train_dialogue_length[i]]=1.0
    val_mask = np.zeros((len(val_dialogue_length), max_utts), dtype='float')
    for i in range(len(val_dialogue_length)):
        val_mask[i,:val_dialogue_length[i]]=1.0
    test_mask = np.zeros((len(test_dialogue_length), max_utts), dtype='float')
    for i in range(len(test_dialogue_length)):
        test_mask[i,:test_dialogue_length[i]]=1.0

    X = (train_dialogue_features, val_dialogue_features, test_dialogue_features)
    y = (train_dialogue_label, val_dialogue_label, test_dialogue_label)
    print(X)
    print(y)
    return X, y
    # assert False

    # train = []
    # val = []
    # test = []
    
    # for i in range(len(data)):
    #     d = data[i]
    #     if d['split'] == 'train':
    #         segment = f"{d['dialog']}_{d['utterance']}"
    #         train.append(
    #             ((d['text'], train_audio_emb[f"{segment}"]),
    #             d['y'], segment)
    #         )
    #     elif d['split'] == 'val':
    #         segment = f"{d['dialog']}_{d['utterance']}"
    #         val.append(
    #             ((d['text'], val_audio_emb[f"{segment}"]),
    #             d['y'], segment)
    #         )
    #     elif d['split'] == 'test':
    #         segment = f"{d['dialog']}_{d['utterance']}"
    #         test.append(
    #             ((d['text'], test_audio_emb[f"{segment}"]),
    #             d['y'], segment)
    #         )

    # return train, val, test

def createOneHot(train_label, test_label):
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test

def get_iemocap_raw(classes):
    f = open("data//IEMOCAP_features.pkl", "rb")
    videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(f)

    '''
    label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
    '''

    #print(videoSentence[trainVid[0]])

    # print(len(trainVid))
    # print(len(testVid))

    train_audio = []
    train_text = []
    train_visual = []
    train_seq_len = []
    train_label = []

    test_audio = []
    test_text = []
    test_visual = []
    test_seq_len = []
    test_label = []

    for vid in trainVid:
        train_seq_len.append(len(videoIDs[vid]))
    for vid in testVid:
        test_seq_len.append(len(videoIDs[vid]))


    max_len = max(max(train_seq_len), max(test_seq_len))
    print('max_len', max_len)
    for vid in trainVid:
        #print(videoLabels[vid])
        train_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        #print(len(train_label[0]))
        #assert False
        print(len(videoText[vid]))
        print(videoSentence[vid])
        #print(len(videoSentence[vid]))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        #print(len(videoAudio[vid]))
        #print(videoVisual[vid][0].shape)

        train_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        train_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        train_visual.append(video)
    assert False
    for vid in testVid:
        test_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        test_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        test_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        test_visual.append(video)

    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)
    train_visual = np.stack(train_visual, axis=0)
    # print(train_text.shape)
    # print(train_audio.shape)
    # print(train_visual.shape)

    # print()
    test_text = np.stack(test_text, axis=0)
    test_audio = np.stack(test_audio, axis=0)
    test_visual = np.stack(test_visual, axis=0)
    # print(test_text.shape)
    # print(test_audio.shape)
    # print(test_visual.shape)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_seq_len = np.array(train_seq_len)
    test_seq_len = np.array(test_seq_len)
    # print(train_label.shape)
    # print(test_label.shape)
    # print(train_seq_len.shape)
    # print(test_seq_len.shape)

    train_mask = np.zeros((train_text.shape[0], train_text.shape[1]), dtype='float')
    for i in range(len(train_seq_len)):
        train_mask[i, :train_seq_len[i]] = 1.0

    test_mask = np.zeros((test_text.shape[0], test_text.shape[1]), dtype='float')
    for i in range(len(test_seq_len)):
        test_mask[i, :test_seq_len[i]] = 1.0

    train_label, test_label = createOneHot(train_label, test_label)

    print(train_text.shape)
    print(train_visual.shape)
    print(train_audio.shape)
    print(train_label.shape)

    train_data = np.concatenate((train_audio, train_visual, train_text), axis=-1)
    test_data = np.concatenate((test_audio, test_visual, test_text), axis=-1)
    print(train_data.shape)
    #print(train_data[0])
    # print(train_data[0][0])
    # print(test_data.shape)

if __name__ == "__main__":
    train, val ,test = parse_MELD()
    #get_iemocap_raw(None)
    save(train, val, test, "meld")