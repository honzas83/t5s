#!/usr/bin/env python3
# coding: utf-8


import argparse
import logging
from t5s import T5

logger = logging.getLogger('t5s.fine_tune')

parser = argparse.ArgumentParser(description='T5 fine-tuner')
parser.add_argument('config', metavar='YAML', type=str, help='Configuration YAML file')

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-10s %(message)s', level=logging.DEBUG)

    args = parser.parse_args()

    t5 = T5(args.config)
    t5.fine_tune()
