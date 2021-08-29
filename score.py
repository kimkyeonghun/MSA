import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def ACC7(value):
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
    return value

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

    return newPreds,newYtest

def main():
    preds = np.load('./predict.npy')
    labels = np.load('./target.npy')
    #print(preds - labels)
    mae_preds = np.tanh(preds)
    mae_labels = np.tanh(labels)
    mae = np.mean(np.absolute(mae_preds - mae_labels))

    y_test7 = ACC7(labels)
    preds7 = ACC7(preds)

    acc7 = accuracy_score(y_test7, preds7)

    preds2 = preds >= 0
    y_test2 = labels >= 0

    acc2 = accuracy_score(y_test2, preds2)
    f_score2 = f1_score(y_test2, preds2, average="weighted")

    preds3, y_test3 = ACC3(preds, labels)

    acc3 = accuracy_score(y_test3, preds3)
    f_score3 = f1_score(y_test3, preds3, average="weighted")

    print("Test_ACC7 : {}, Test_ACC3 : {}, Test_ACC2 : {}, Test_MAE : {}, Test_F_Score2: {}, Test_F_Score3: {}".format(acc7,acc3,acc2,mae,f_score2, f_score3))
if __name__ == '__main__':
    main()