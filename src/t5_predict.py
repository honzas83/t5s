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

    batch_size = t5.config.get("predict", {}).get("batch_size", 400)

    with open(args.tsv_in, "r", encoding="utf-8") as fr, \
         open(args.tsv_out, "w", encoding="utf-8") as fw:

        batch = []

        def flush(n_predicted=[0]):
            preds = t5.predict(batch)

            for input_sent, output_sent in zip(batch, preds):
                n_predicted[0] += 1
                print(input_sent, output_sent, sep="\t", file=fw)
                fw.flush()
                
            del batch[:]
            logger.info("Processed %d items", n_predicted[0])

        for line in fr:
            items = line.strip().split("\t")
            input_sent = items[0]
            batch.append(input_sent)
            if len(batch) >= batch_size:
                flush()
        else:
            if batch:
                flush()
