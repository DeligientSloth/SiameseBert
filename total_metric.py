import sys,os,re
import json
import pandas as pd
from commen_function import *
from sklearn import metrics
from functools import reduce

input_path = sys.argv[1]
#output_path = sys.argv[2]
thr = float(sys.argv[2])

d = {}  # 字典，保存一个query下的所有question
for line in open(input_path):
    q1, q2, label, predict = line.strip().split(",")
    if q1 == "user_query":  #  csv文件的头部
        continue
    d.setdefault(q1, [])
    d[q1].append((q2, label, predict))

#f = open(output_path, "w")
P, N, TP, FN, FP, TN = 0,0,0,0,0,0
for q1 in d:
    terms = d[q1]
    
    sort_l = [(predict, (q2, label, predict)) for q2, label, predict in terms] # 针对每一个query的所有question预测结果排序
    sort_l.sort(reverse=True)
    terms_str = "|".join(["-".join(list(term[1])) for term in sort_l])
    terms_str = str(sort_l)

    best_term = sort_l[0][1]  #  预测最高的
    q2, label, predict = best_term

    predict_label = 1 if float(predict) > thr else 0 # best answer的结果 是否过阈值
    #predict_answer = predict_label * int(label) # 判定best answer预测结果是否正确
    has_answer = int(any([int(label) for q2, label, predict in terms])) # 判定候选question是否有正确答案

    # predict_answer has_answer  q1 q2_list 作为最终的判定序列
    #f.write("\t".join([str(has_answer), label, str(predict_label), predict, q1, terms_str]) + "\n")
    #  都是从query出发的
    label = int(label)
    if has_answer == 1 : P += 1 # 正类 候选包含正缺答案，也就是query至少跟一个question匹配上，
    if has_answer == 0 : N += 1 # 负类 候选不包含正确答案，query没有question可以匹配到，体现query是负的
    if label == 1 and predict_label == 1: TP += 1 # 将正类预测为正类数
    if label == 1 and predict_label == 0: FN += 1 # 将正类预测为负类数
    if label == 0 and predict_label == 1: FP += 1 # 将负类预测为正类
    if label == 0 and predict_label == 0: TN += 1 # 将负类预测为负类数

PRE = TP / (TP + FP)
REC = TP / P
ACC = (TP + TN) / (P + N)
F1  = 2 / (1/PRE + 1/REC)

print ("Query级别结果指标")
print ("THR={} P={} N={} TP={} FN={} FP={} TN={}".format(thr, P, N, TP, FN, FP, TN))
print ("ACC : %.4f" % ACC)
print ("PRE : %.4f" % PRE)
print ("REC : %.4f" % REC)
print ("F1  : %.4f" % F1)


###################################################################
f = pd.read_csv(input_path, sep = ',')

f.loc[f.score > thr, 'pred'] = 1
f.loc[f.score <= thr, 'pred'] = 0
print ()
print ("Pair级别结果指标")
print(f.describe())
print("ACC::",metrics.accuracy_score(f.label, f.pred))
print("PRE::",metrics.precision_score(f.label, f.pred))
print("REC::",metrics.recall_score(f.label, f.pred))
print("F1:::",metrics.f1_score(f.label, f.pred))

def skl_get_auc(tag_input, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(tag_input, predictions, pos_label=1)
    return metrics.auc(fpr, tpr,reorder=True)

auc = skl_get_auc(f.label, f.score)
print("AUC::", auc)
