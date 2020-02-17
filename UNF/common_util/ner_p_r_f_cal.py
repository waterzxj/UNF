#coding:utf-8
import os
import sys

sys.path.append("../")

from training.learner_util import get_ner_BIO


gold_loc_tp = 0.0
gold_loc_fp = 0.0
gold_loc_tn = 0.0
gold_loc_fn = 0.0

gold_per_tp = 0.0
gold_per_fp = 0.0
gold_per_tn = 0.0
gold_per_fn = 0.0

gold_org_tp = 0.0
gold_org_fp = 0.0
gold_org_tn = 0.0
gold_org_fn = 0.0


for line in open("../model_lstm/test_ner"):
    line = line.rstrip()
    parts = line.split("\t")
    query = parts[0]
    labels = get_ner_BIO(parts[1].split())
    preds = get_ner_BIO(parts[2].split())
    if labels or preds:
        for item in labels:
            if item in preds:
                if "LOC" in item:
                    gold_loc_tp += 1
                elif "ORG" in item:
                    gold_org_tp += 1
                elif "PER" in item:
                    gold_per_tp += 1

            else:
                if "LOC" in item:
                    gold_loc_fn += 1
                elif "ORG" in item:
                    gold_org_fn += 1
                elif "PER" in item:
                    gold_per_fn += 1

        for item in preds:
            if item not in labels:
                if "LOC" in item:
                    gold_loc_fp += 1
                elif "ORG" in item:
                    gold_org_fp += 1
                elif "PER" in item:
                    gold_per_fp += 1


loc_pre = gold_loc_tp * 1.0 / (gold_loc_tp + gold_loc_fp)
loc_rec = gold_loc_tp * 1.0 / (gold_loc_tp + gold_loc_fn)
loc_f = 2 * loc_pre * loc_rec / (loc_pre + loc_rec)
print("Location precision:%s recall:%s f:%s" % (loc_pre, loc_rec, loc_f))


per_pre = gold_per_tp * 1.0 / (gold_per_tp + gold_per_fp)
per_rec = gold_per_tp * 1.0 / (gold_per_tp + gold_per_fn)
per_f = 2 * per_pre * per_rec / (per_pre + per_rec)
print("per precision:%s recall:%s f:%s" % (per_pre, per_rec, per_f))

org_pre = gold_org_tp * 1.0 / (gold_org_tp + gold_org_fp)
org_rec = gold_org_tp * 1.0 / (gold_org_tp + gold_org_fn)
org_f = 2 * org_pre * org_rec / (org_pre + org_rec)
print("org precision:%s recall:%s f:%s" % (org_pre, org_rec, org_f))
                


    
    


