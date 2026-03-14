import numpy as np
import pandas as pd
from util.TSB_AD.metrics import metricor
from util.TSB_AD.slidingWindows import find_length #,plotFig, printResult
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import os
import sys

import math
import arff
import time
import pickle

from util.TSB_AD.models.norma import NORMA
from util.TSB_AD.models.sand import SAND
from util.TSB_AD.models.damp import DAMP

import warnings
warnings.filterwarnings('ignore')

def running_mean(x,N):
	return (np.cumsum(np.insert(x,0,0))[N:] - np.cumsum(np.insert(x,0,0))[:-N])/N

# slidingWindow=100
def compare_methods(data, label, slidingWindow, init_chunk=5000, batch=2000):
    # slidingWindow = find_length(data)
    print('SlidingWindow:', slidingWindow)

    x_test = data
    scores = []
    slabels = []
    
    
    process_time = []

    print('Test: NORMA-OFF')
    start_t = time.time()
    clf_off = NORMA(pattern_length = slidingWindow, nm_size=3*slidingWindow, percentage_sel=1, normalize=True)
    clf_off.fit(x_test)
    end_t = time.time()
    process_time.append(end_t -start_t)
    print('NormA-Done (takes)', end_t - start_t)
    score = clf_off.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*((slidingWindow-1)//2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    scores.append(score[:len(x_test)])
    slabels.append('NormA (off)\nAnomaly Score')

    print(len(scores))

    modelName='SAND (online)'
    start_t = time.time()
    clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
    x = data
    clf.fit(x,online=True,alpha=0.5,init_length=init_chunk,batch_size=batch,verbose=True,overlaping_rate=int(4*slidingWindow))
    end_t = time.time()
    process_time.append(end_t -start_t)
    print('SAND-Done (takes)', end_t - start_t)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    scores.append(score[:len(x_test)])
    slabels.append('SAND (online)\nAnomaly Score')

    print(len(scores))

    time_df = pd.DataFrame(process_time, columns=['time'])
    # display(time_df)
    return scores, slabels

def read_arff(filename):
    """
    Find ndarray corresponding to data and labels from arff data
    """
    arff_content = arff.load(f.replace(',\n', '\n') for f in open(filename, 'r'))
    arff_data = arff_content['data']
    data = np.array([i[:1] for i in arff_data])
    anomaly_labels = np.array([i[-1] for i in arff_data])
    anomaly_labels = anomaly_labels.reshape((len(anomaly_labels),1))
    return data.astype(float), anomaly_labels.astype(float)

peak_adj_columns = ['AUC', 'Precision', 'Recall', 'F1', 'TH', 'RPrecision', 'RRecall', 'RF1', 'PaK', 'F1_adj', 'Precision_adj', 'Recall_adj', 'roc_auc_adj']
###################################################################
## The source of 'adjust_predict' function is 'TranAD' github repo.
## https://github.com/imperial-qore/TranAD.git
# 
# the below function is taken from OmniAnomaly code base directly
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    print(type(score), score.dtype, threshold)
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

###################################################################
## The source of 'calc_point2point' function is 'TranAD' github repo.
## https://github.com/imperial-qore/TranAD.git
# 
def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = metrics.roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc

def peakf1_acc(label, score, plot_AUC=False, alpha=0.2):
    grader = metricor()
    result = pd.DataFrame(columns=peak_adj_columns)
    if np.sum(label) != 0:
        auc = metrics.roc_auc_score(label, score)

        # plor ROC curve
        fpr, tpr, _ = metrics.roc_curve(label, score)
        pr, re, thresholds = metrics.precision_recall_curve(label, score)

        if plot_AUC:
            dp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            dp.plot()
        
        peak_f1, peak_ind = np.nanmax(2*(pr*re)/(pr+re)), np.nanargmax(2*(pr*re)/(pr+re))

        peak_ths = thresholds[peak_ind] 

        #range anomaly 
        preds = score > peak_ths
        Rrecall, ExistenceReward, OverlapReward = grader.range_recall_new(label, preds, alpha)
        Rprecision = grader.range_recall_new(preds, label, 0)[0]


        if Rprecision + Rrecall==0:
            Rf=0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)


        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1-k/len(label)))


        p_at_k = np.where(score > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k/k


        ## Adjustment csae
        pred_adj = adjust_predicts(score, label,
                threshold=peak_ths,
                pred=None,
                calc_latency=False)
        
        f1_adj, pr_adj, re_adj, _, _, _, _, auc_adj = calc_point2point(pred_adj, label)


        result.loc[0] = [auc, pr[peak_ind], re[peak_ind], peak_f1, peak_ths, Rprecision, Rrecall, Rf, precision_at_k, f1_adj, pr_adj, re_adj, auc_adj]
        return result
    
def result_f1_acc(methods, scores, label, ths = None):
    result_org = pd.DataFrame(columns=['method'] + peak_adj_columns)
    j = 0
    for i, method in enumerate(methods):
        # r_tmp = get_acc(label.reshape(-1)[:len(scores[i])], np.array(scores[i]), slidingWindow, ths)
        r_tmp = peakf1_acc(label.reshape(-1)[:len(scores[i])], np.array(scores[i]), plot_AUC=False)
        if r_tmp is not None:
            result_org.loc[j] = [method] + list(r_tmp.loc[0])
        else:
            result_org.loc[j] = [method] + [0]*len(peak_adj_columns)
        j+=1


    # display(result_org)
    return result_org

data_folders = []
data_folders.append('/data/test_weather/org')
data_folders.append('/data/test_weather/n_drift')
data_folders.append('/data/test_weather/p_drift')
data_folders.append('/data/test_weather_mod/org')
data_folders.append('/data/test_weather_mod/n_drift')
data_folders.append('/data/test_weather_mod/p_drift')

init_chunk = 24*60
batch = 24*28

def main(argv):
    print('\n')
    print('argv: ', argv)

    ### Run the original datasets
    for dir_data in data_folders:
        dir = os.getcwd() + dir_data
        all_files = os.listdir(dir)
        filenames = [file for file in all_files if file.endswith('.arff')]
        print(filenames)
        org_scores = []
        org_slabels = []
        for f in filenames:
            data, label = read_arff(f'{dir}/{f}')
            data = data.reshape(-1)
            label = label.reshape(-1)
            slidingWindow = 24
            scores, slabels = compare_methods(data, label, slidingWindow, init_chunk, batch)
            org_scores.append(scores)
            org_slabels.append(slabels)


        name_f = dir_data.split('/')[-1]
        with open(f'{dir}/result_score_{name_f}_2.pkl', 'wb') as f:
            pickle.dump(org_scores, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    argv = sys.argv
    main(argv)