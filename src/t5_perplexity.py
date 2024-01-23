#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import argparse
import logging
from pprint import pprint
import yaml
from t5s import T5

logger = logging.getLogger('t5s.perplexity')

parser = argparse.ArgumentParser(description='T5 perplexity computation')
parser.add_argument('config', metavar='YAML', type=str, help='Configuration YAML file')
parser.add_argument('tsv_in', metavar='TSV_IN', nargs='+', type=str, help='Input TSV file - for perplexity, the targets must be specified.')
parser.add_argument('--skip-first', metavar='N', type=int, default=0, help='Skip first N tokens while computing PPL.')
parser.add_argument('--skip-last', metavar='M', type=int, default=0, help='Skip last M tokens while computing PPL.')
parser.add_argument('--round-digits', metavar='K', type=int, default=3, help='Round PPL to K digits.')
parser.add_argument('--batch-size', metavar='B', type=int, default=None, help='Overwrite batch_size from config YAML')

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-10s %(message)s', level=logging.DEBUG)

    args = parser.parse_args()

    t5 = T5(args.config)
    if args.skip_first == 0 and args.skip_last == 0:
        slice_description = ""
    else:
        slice_description = f"[{args.skip_first if args.skip_first != 0 else ''}:{-args.skip_last if args.skip_last != 0 else ''}]"

    outputs = []
    for fn in args.tsv_in:
        logger.info(f"Processing file: {fn}")
        losses = t5.compute_perplexity_tsv(fn, batch_size=args.batch_size)

        flat_losses = [item for row in losses for item in row[args.skip_first:len(row)-args.skip_last]]
        ppl = 2**(sum(flat_losses)/len(flat_losses))
        ppl = round(ppl, args.round_digits)

        outputs.append(f"{args.config}: {fn}: perplexity{slice_description}: {ppl}")

    for output in outputs:
        print(output)
