#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging
import numpy as np
import json
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger('eval_tsv')

parser = argparse.ArgumentParser(description='Evaluates TSV files')
parser.add_argument('metric', metavar='METRIC', type=str, help='Metric')
parser.add_argument('ref', metavar='REF', type=str, help='Reference TSV')
parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis TSV')


def f1_multilabel(pairs):
    def slash_split(item):
        return tuple(i.strip() for i in item.split("/"))

    ref, hyp = zip(*pairs)
    ref = [slash_split(i) for i in ref]
    hyp = [slash_split(i) for i in hyp]

    kwds_set = set()
    for lst in [ref, hyp]:
        for kws in lst:
            kwds_set |= set(kws)

    kwds_list = {kw: idx for idx, kw in enumerate(list(kwds_set))}

    def to_array(lst):
        ret = np.zeros((len(lst), len(kwds_list)))
        for idx, item in enumerate(lst):
            for kw in item:
                ret[idx, kwds_list[kw]] = 1
        return ret

    ref = to_array(ref)
    hyp = to_array(hyp)

    P, R, F, _ = precision_recall_fscore_support(ref, hyp, average="samples")
    return {"P": P, "R": R, "F": F}


def match(pairs):
    n = 0
    ok = 0
    w_n = 0
    w_ok = 0
    for ref, hyp in pairs:
        if ref == hyp:
            ok += 1
        n += 1

        ref = ref.split()
        hyp = hyp.split()
        w_n += len(ref)
        for r1, h1 in zip(ref, hyp):
            if r1 == h1:
                w_ok += 1

    return {"SAcc": ok/n, "WAcc": w_ok/w_n, "W_N": w_n, "W_OK": w_ok, "S_N": n, "S_OK": ok, "W_Err": w_n-w_ok, "S_Err": n-ok}


def binary_lab(pairs):
    TP = 0
    FN = 0
    FP = 0
    for ref, hyp in pairs:
        ref = ref.split()
        hyp = hyp.split()
        for r, h in zip(ref, hyp):
            if r == h == "1":
                TP += 1
            elif r == "1" and h == "0":
                FN += 1
            elif r == "0" and h == "1":
                FP += 1

    P = TP / (TP+FP)
    R = TP / (TP+FN)
    F = 2 * P * R / (P+R)

    return {"TP": TP, "FN": FN, "FP": FP, "P": P, "R": R, "F": F}


METRICS = {
    "f1_multilabel": f1_multilabel,
    "match": match,
    "binary_lab": binary_lab,
}


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-10s %(message)s', level=logging.DEBUG)

    args = parser.parse_args()

    pairs = []

    metric = METRICS[args.metric]

    with open(args.ref, "r", encoding="utf-8") as fr_ref, \
         open(args.hyp, "r", encoding="utf-8") as fr_hyp:
        for ref_line, hyp_line in zip(fr_ref, fr_hyp):
            ref_in, ref_out = ref_line.split("\t")[:2]
            hyp_in, hyp_out = hyp_line.split("\t")[:2]

            pairs.append((ref_out, hyp_out))

        logger.info("Loaded %d examples", len(pairs))

    ret = metric(pairs)

    print(json.dumps(ret, indent=4, sort_keys=True))
