#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging
from t5s import EVAL_METRICS, eval_tsv, yaml_dump_result
import sys

logger = logging.getLogger('eval_tsv')

parser = argparse.ArgumentParser(description='Evaluates TSV files')
parser.add_argument('metric', metavar='METRIC', choices=sorted(EVAL_METRICS.keys()), type=str, help='Metric to evaluate')
parser.add_argument('ref', metavar='REF', type=str, help='Reference TSV')
parser.add_argument('hyp', metavar='HYP', nargs="+", type=str, help='Hypothesis TSV')


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-10s %(message)s', level=logging.DEBUG)

    args = parser.parse_args()

    pairs = []

    metric = EVAL_METRICS[args.metric]

    ret = eval_tsv(metric, args.ref, args.hyp)
    yaml_dump_result(ret, sys.stdout)
