#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging
from t5s import T5

parser = argparse.ArgumentParser(description="Executes T5 evaluation")
parser.add_argument("config", metavar="YAML", type=str, help="Configuration YAML file")
parser.add_argument("datasets", metavar="DATASET", nargs="*", default=None, type=str,
                    help="Perform evaluation on these datasets (overrides configuration option evaluation.datasets).")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-10s %(message)s", level=logging.DEBUG)

    args = parser.parse_args()

    t5 = T5(args.config)
    if not args.datasets:
        args.datasets = None
    t5.evaluate(args.datasets)
