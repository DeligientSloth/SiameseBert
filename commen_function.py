#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import traceback
from sklearn import metrics

import io
import codecs

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')




### common--funciton ##############################################
def add_color(s):
    cl = "\033[1;31;40m"
    cr = "\033[0m"
    return cl + str(s) + cr

def vec_add_color(vec, idx):
    s = vec[idx]
    vec[idx] = add_color(s)

def warning(message):
    sys.stderr.write(add_color(message))
    sys.stderr.write("\n")

def WARNING(message):
    warning(message)

def fatal(e):
    if type(e) == str:
        warning(e)
    else:
        warning("*"*64)
        #warning("%s%s" % ('str(Exception):\t', str(E)))
        warning("%s%s" % ('str(e):\t\t', str(e)))
        warning("%s%s" % ('repr(e):\t', repr(e)))
        #warning("%s%s" % ('e.message:\t', e.message))
        warning("%s%s" % ('traceback.print_exc():', traceback.print_exc()))
        #warning("%s%s" % ('traceback.format_exc():', traceback.format_exc()))
        warning("*"*64)

def FATAL(e):
    fatal(e)

### conv--fcuntion ###############################################

def list_conv(l, func):
    return [func(x) for x in l]

def list_filter(l, func):
    return [x for x in l if func(x)]

def list_pad(l, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        max_seq_len: max len for padding
    Returns:
        a list of list where each sublist has same length
    """
    result = [l[x] if x < len(l) else pad_tok for x in range(max_length)]
    return result

def list_tuple_2_lists(tuple_list):
    a,b,c = [],[],[]
    for x in tuple_list:
        a.append(x[0])
        b.append(x[1])
        c.append(x[2])
    return a,b,c

def list_2_tuple_2_lists(tuple_list):
    a,b = [],[]
    for x in tuple_list:
        a.append(x[0])
        b.append(x[1])
    return a,b

def remove_stopword(l, stop_word_d):
    return list_filter(l, lambda x: x not in stop_word_d)

### file--funciton ##################################################

def code_conv(msg, decode=None, encode=None):
    if not decode is None : msg = msg.decode(decode)
    if not encode is None : msg = msg.encode(encode)
    return msg

def load_word_index(input_f, decode=None, encode=None):
    """
    load input_f per line for a index in dict.
    """
    d = {}
    with open(input_f, encoding="utf-8") as f:
        for line in f:
            line = code_conv(line.strip(), decode=decode, encode=encode)
            if line not in d:
                d.setdefault(line, len(d) + 1)
    return d

def load_file_2_list(input_f, strip_ch=None, decode=None, encode=None, filter=None, limit=None):
    """
    input_f: the file of input
    decode: the coding to decode. eg: utf-8
    encode: the coding to encode. eg: utf-8
    filter: the function of filter.  eg: lambda x: len(x) > 1
    limit: the num of lines to load
    """
    ans_l = []
    i = 0
    with open(input_f, encoding="utf-8") as f:
        for line in f:
            msg = line.strip() if strip_ch is None else line.strip(strip_ch)
            msg = code_conv(msg, decode=decode, encode=encode)
            ans_l.append(msg)
            i+=1
            if not limit is None and i % limit == 0:
                break
    return ans_l

def load_word2vec(input_f, filter_num=None, limit=None, decode=None, encode=None):
    """
    input_f: the file of input
    filter_num: if the num of the seg for a segment line not is filter_num, then filter it
    """
    d = {}
    for tokens in load_file_2_list(input_f, limit=limit):
        tokens = tokens.split(" ")
        if filter_num != None and len(tokens) != filter_num + 1:
            continue
        k = code_conv(tokens[0], decode=decode, encode=encode)
        v = list_conv(tokens[1:], lambda x: float(x))
        d.setdefault(k,v)
    return d

def load_file_2_dict(input_f, judge=False, colum=2, value_default=1, decode=None, encode=None):
    """
    input_f: the file of input.
    judge: if the dict if judge dict or not.
        True: make the dict value is value_default.
        False: make the dict value is the last colum value.
    colum: the columu num, not the dict hierarchy.
    value_default: the dict value when the colum is 1 or the judge is True.
    """
    d = {}
    with open(input_f, encoding="utf-8") as f:
        for line in f:
            line = code_conv(line.strip(), decode=decode, encode=encode)
            tokens = line.split("\t")[:colum]
            if (len(tokens) != colum):
                continue
            if len(tokens) == 1:
                d.setdefault(tokens[0], value_default)
            elif len(tokens) == 2:
                k,v = tokens
                if judge==True:
                    d.setdefault(k, {})
                    d[k].setdefault(v, value_default)
                else :
                    d.setdefault(k, v)
            elif len(tokens) == 3:
                k,v,e = tokens
                d.setdefault(k, {})
                if judge==True:
                    d[k].setdefault(v, {})
                    d[k][v].setdefault(e,value_default)
                else :
                    d[k].setdefault(v, e)
            else:
                continue
    return d

#####  embedding-function ##################################3


### evaluation—-funciton ############################################

def skl_get_auc(tag_input, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(tag_input, predictions, pos_label=1)
    return metrics.auc(fpr, tpr,reorder=True)

def skl_get_rec(tag_input, predictions):
    return metrics.recall_score(tag_input, predictions)

def skl_get_f1 (tag_input, predictions):
    return metrics.f1_score(tag_input, predictions)

def skl_get_acc(tag_input, predictions):
    return metrics.accuracy_score(tag_input, predictions)

def calc_target(labels,predicts):
    if len(labels) != len(predicts):
        WARNING("labels and predicts length not match!")

    TP = 0.0 # 将正类预测为正类数
    FN = 0.0 # 将正类预测为负类数
    FP = 0.0 # 将负类预测为正类
    TN = 0.0 # 将负类预测为负类数
    P  = 0.0 # 正类
    N  = 0.0 # 负类

    for i in range(len(labels)):
        l,p = labels[i], predicts[i]
        if l == 1 and p == 1 : TP += 1
        if l == 1 and p == 0 : FN += 1
        if l == 0 and p == 1 : FP += 1
        if l == 0 and p == 0 : TN += 1

        if l == 1 : P += 1
        if l == 0 : N += 1

    PRE = TP / (TP + FP)
    REC = TP / (TP + FN)
    ACC = (TP + TN) / (P + N)
    F1  = 2*TP / (2*TP + FP + FN)
    F2  = 2 / (1/PRE + 1/REC)

    return TP, FN, FP, TN, ACC, PRE, REC, F1, F2
