#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import argparse
import logging
from pprint import pprint
import yaml
from t5s import T5

logger = logging.getLogger('t5s.fine_tune')

parser = argparse.ArgumentParser(description='T5 fine-tuner')
parser.add_argument('config', metavar='YAML', type=str, help='Configuration YAML file')
parser.add_argument('tsv_in', metavar='TSV_IN', type=str, help='Input TSV file - could be a simple TXT file without targets.')
parser.add_argument('tsv_out', metavar='TSV_OUT', type=str, help='Output TSV file')

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-10s %(message)s', level=logging.DEBUG)

    args = parser.parse_args()

    t5 = T5(args.config)
    t5.predict_tsv(args.tsv_in, args.tsv_out)

