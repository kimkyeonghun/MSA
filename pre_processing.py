from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
import sys

import os
import re
import mmsdk
import pickle
import argparse
import numpy as np
from tqdm import tqdm_notebook
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError


def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features

def download_dataset(datasetName):
    if SDK_PATH is None:
        print("SDK path is not specified! Please specify first in constants/paths.py")
        exit(0)
    else:
        sys.path.append(SDK_PATH)

    if not os.path.exists(DATA_PATH):
        check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

    if datasetName == "cmu_mosi":
        DATASET = md.cmu_mosi
    elif datasetName == "cmu_mosei":
        DATASET = md.cmu_mosei

    print(DATASET.highlevel)

    try:
        md.mmdataset(DATASET.highlevel,DATA_PATH)
    except RuntimeError:
        print("High-level features have been donwloaded previously.")

    try:
        md.mmdataset(DATASET.raw,DATA_PATH)
    except RuntimeError:
        print("Raw data have been downloaded previously.")

    try:
        md.mmdataset(DATASET.labels, DATA_PATH)
    except RuntimeError:
        print("Labels have been downloaded previously.")

    TRAINSPLIT = DATASET.standard_folds.standard_train_fold
    VALSPLIT = DATASET.standard_folds.standard_valid_fold
    TESTSPLIT = DATASET.standard_folds.standard_test_fold

    return TRAINSPLIT, VALSPLIT, TESTSPLIT



def prepare_save(features, dataset, TRAINSPLIT, VALSPLIT, TESTSPLIT):

    textField, visualField, speechField, labelField = features
    train = []
    val = []
    test = []
    
    EPS = 0

    # define a regular expression to extract the video ID out of the keys
    pattern = re.compile('(.*)\[.*\]')
    num_drop = 0 # a counter to count how many data points went into some processing issues

    for segment in dataset[labelField].keys():
        
        # get the video ID and the features out of the aligned dataset
        vid = re.search(pattern, segment).group(1)
        try:
            label = dataset[labelField][segment]
            _words = dataset[textField][segment]
            _visual = dataset[visualField][segment]
            _speech = dataset[speechField][segment]
        except:
            print(f"[Segment]Found video that doesn't belong to any splits: {segment}")
            num_drop +=1
            continue
            
        label = label['features']
        _words = _words['features']
        _visual = _visual['features']
        _speech = _speech['features']

        # if the sequences are not same length after alignment, there must be some problem with some modalities
        # we should drop it or inspect the data again
        if not _words.shape[0] == _visual.shape[0] == _speech.shape[0]:
            print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_speech.shape}")
            num_drop += 1
            continue

        # remove nan values
        label = np.nan_to_num(label)
        _visual = np.nan_to_num(_visual)
        _speech = np.nan_to_num(_speech)

        # remove speech pause tokens - this is in general helpful
        # we should remove speech pauses and corresponding visual/acoustic features together
        # otherwise modalities would no longer be aligned
        words = []
        visual = []
        speech = []
        for i, word in enumerate(_words):
            if word[0] != b'sp':
                words.append(word[0].decode('utf-8')) # SDK stores strings as bytes, decode into strings here
                visual.append(_visual[i, :])
                speech.append(_speech[i, :])

        words = np.asarray(words)
        visual = np.asarray(visual)
        speech = np.asarray(speech)

        # z-normalization per instance and remove nan/infs
        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
        speech = np.nan_to_num((speech - speech.mean(0, keepdims=True)) / (EPS + np.std(speech, axis=0, keepdims=True)))

        if vid in TRAINSPLIT:
            train.append(((words, visual, speech), label, segment))
        elif vid in VALSPLIT:
            val.append(((words, visual, speech), label, segment))
        elif vid in TESTSPLIT:
            test.append(((words, visual, speech), label, segment))
        else:
            print(f"[Videosplit]Found video that doesn't belong to any splits: {vid}")

    print(f"Total number of {num_drop} datapoints have been dropped.")

    return train, val, test

def save(train, val, test, datasetName):
    allDataset = {"train":train, "val" : val, "test": test}
    with open('{}.pkl'.format(datasetName),'wb') as f:
        pickle.dump(allDataset,f)
    print("Save Complete!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--textField",type=str,default='CMU_MOSI_ModifiedTimestampedWords')
    parser.add_argument("--visualField",type=str,default='CMU_MOSI_Visual_Facet_41')
    parser.add_argument("--speechField",type=str,default='CMU_MOSI_COVAREP')
    parser.add_argument("--labelField",type=str,default='CMU_MOSI_Opinion_Labels')
    parser.add_argument("--datasetName",type=str,required=True)

    args = parser.parse_args()

    TRAINSPLIT, VALSPLIT, TESTSPLIT = download_dataset(args.datasetName)

    features = [
        args.textField,
        args.visualField,
        args.speechField
    ]
    recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
    
    dataset = md.mmdataset(recipe)
    dataset.align(args.textField, collapse_functions=[avg])

    labelRecipe = {args.labelField: os.path.join(DATA_PATH, args.labelField + '.csd')}
    dataset.add_computational_sequences(labelRecipe, destination=None)
    dataset.align(args.labelField)

    features.append(args.labelField)

    train, val, test = prepare_save(features, dataset,  TRAINSPLIT, VALSPLIT, TESTSPLIT)

    save(train, val, test, args.datasetName)

if __name__ == "__main__":
    main()