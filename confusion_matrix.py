import numpy as np

def calcEvaluation(tp, fp, fn, tn):
    # accuracy
    if tp + tn + fp + fn == 0:
        accuracy = -1
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy = np.round(accuracy * 100, 2)
    # recall
    if tp + fn == 0:
        recall = -1
    else:
        recall = tp / (tp + fn)
        recall = np.round(recall * 100, 2)
    # precition
    if tp + fp==0:
        precition = -1
    else:
        precition = tp / (tp + fp)
        precition = np.round(precition * 100, 2)
    # F1
    if 2 * tp + fp + fn==0:
        f1 = -1
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
        f1 = np.round(f1 * 100, 2)
    
    return accuracy, recall, precition, f1

def getConfusionMatrix(y_pred, y_true, classes = 3):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    num_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=float)

    for i in range(len(y_true)):
        true_index = y_true[i]
        pred_index = y_pred[i]
        confusion_matrix[true_index][pred_index] += 1
    
    Accuracy = []
    Recall = []
    Precition = []
    F1 = []

    cm = confusion_matrix

    for index in range(classes):
        l = np.delete(np.arange(classes), index)
        
        tp = cm[index][index]

        fp = np.delete(np.delete(cm, index, axis=0), l, axis=1)
        fp = np.sum(fp)

        fn = np.delete(np.delete(cm, l, axis=0), index, axis=1)
        fn = np.sum(fn)

        tn = np.delete(np.delete(cm, index, axis=0), index, axis=1)
        tn = np.sum(tn)

        accuracy, recall, precition, f1 = calcEvaluation(tp, fp, fn, tn)

        Accuracy.append(accuracy)
        Recall.append(recall)
        Precition.append(precition)
        F1.append(f1)

    return Accuracy, Recall, Precition, F1
