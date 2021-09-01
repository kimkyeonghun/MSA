import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def ACC7(value, true):
    """
    for 7 label
    """
    for i,v in enumerate(value):
        if v < -2:
            value[i] = -3
        elif -2 <= v < -1:
            value[i] = -2
        elif -1 <= v < 0:
            value[i] = -1
        elif v==0:
            value[i] = 0
        elif 0 < v <= 1:
            value[i] = 1
        elif 1 < v <= 2:
            value[i] = 2
        elif v > 2:
            value[i] = 3

    for i,v in enumerate(true):
        if v < -2:
            true[i] = -3
        elif -2 <= v < -1:
            true[i] = -2
        elif -1 <= v < 0:
            true[i] = -1
        elif v==0:
            true[i] = 0
        elif 0 < v <= 1:
            true[i] = 1
        elif 1 < v <= 2:
            true[i] = 2
        elif v > 2:
            true[i] = 3
    return np.sum(value==true)/float(len(true))

def ACC3(preds,y_test):
    """
    for 2 label except 0
    """
    newPreds = []
    newYtest = []
    for i,(p,y) in enumerate(zip(preds,y_test)):
        if y > 0:
            newPreds.append(1)
            if p > 0:
                newYtest.append(1)
            else:
                newYtest.append(0)
        elif y < 0:
            newPreds.append(0)
            if p > 0:
                newYtest.append(1)
            else:
                newYtest.append(0)

    return np.array(newPreds),np.array(newYtest)

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def MISA(test_truth,test_preds):
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds = test_preds.squeeze(-1)

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    my_a7 = ACC7(test_preds, test_truth)
    
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    
    # pos - neg
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    tt = sum((binary_preds==1)&(binary_truth==1))
    ft = sum((binary_preds==1)&(binary_truth==0))
    tf = sum((binary_preds==0)&(binary_truth==1))
    ff = sum((binary_preds==0)&(binary_truth==0))

    table = [[tt,tf],[ft,ff]]

    from statsmodels.stats.contingency_tables import mcnemar
    print(mcnemar(table, exact=False))

    if True:
        print("\n")
        print("mae: ", mae)
        print("corr: ", corr)
        print("mult_acc: ", mult_a7)
        print("My ACC7: ", my_a7)
        print("\nClassification Report (pos/neg) :")
        print(classification_report(binary_truth, binary_preds, digits=3))
        print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
    
    # non-neg - neg
    binary_truth = (test_truth >= 0)
    binary_preds = (test_preds >= 0)

    if True:
        print("Classification Report (non-neg/neg) :")
        print(classification_report(binary_truth, binary_preds, digits=3))
        print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str)
    args = parser.parse_args()

    preds = np.load(os.path.join('numpy_save',args.path,'predict.npy'))
    labels = np.load(os.path.join('numpy_save',args.path,'target.npy'))
    print(np.unique(np.round(preds)))
    print(np.unique(np.round(labels)))

    MISA(labels,preds)
if __name__ == '__main__':
    main()