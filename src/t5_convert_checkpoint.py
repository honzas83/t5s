#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import argparse
import logging
from pprint import pprint
import re
from t5s import T5Training, TFT5ForConditionalGeneration
import tensorflow as tf

logger = logging.getLogger('t5s.convert_checkpoint')

parser = argparse.ArgumentParser(description='Converts pre-trained T5 TF checkpoint into huggingface transformers format')
parser.add_argument('base', metavar='BASE', type=str, help='Base model')
parser.add_argument('checkpoint', metavar='CHECKPOINT', type=str, help='TF checkpoint')
parser.add_argument('output', metavar='OUTPUT', type=str, help='Store output model into this directory')


def load_checkpoint(self, checkpoint, assign=False):
        def map_variable(v):
            transpose = "relative_attention_bias" in v
            v = v.replace("rms_norm", "layer_norm")
            v = v.replace("coder/layer_norm", "coder/final_layer_norm")
            m = {
                "shared/embedding": "shared/shared/weight:0",
                "encoder/final_layer_norm/scale": prefix+"/encoder/final_layer_norm/weight:0",
                "decoder/final_layer_norm/scale": prefix+"/decoder/final_layer_norm/weight:0",
                "global_step": None,
            }
            if v in m:
                return m[v], transpose
            elif "_slot_" in v:
                return None, transpose
            else:
                v = re.sub(r"_(\d+)",
                           lambda match: "_._"+str(int(match.group(1))),
                           v
                          )
                if "layer_norm" in v:
                    v = v.replace("/scale", "") + "/weight:0"
                elif "relative_attention_bias" in v:
                    v += "/embeddings:0"
                else:
                    v = v.replace("/kernel", "") + "/kernel:0"
                v = prefix+"/"+v
                return v, transpose
        
        names = [w.name for w in self.weights]
        done_idxs = set()
        prefix = "t5_training"
        logger.info("Common prefix: %s", prefix)
        weights = [None for w in self.weights]
        
        var_list = tf.train.list_variables(checkpoint)

        errors = False

        for tf_var, shape in var_list:
            var, transpose = map_variable(tf_var)
            if var is None:
                continue
            logger.info("Mapping %s --> %s", tf_var, var)
            try:
                idx = names.index(var)
            except ValueError:
                logger.error("Cannot match TF variable: %s", var)
                errors = True
                continue
            tf_value = tf.train.load_variable(checkpoint, tf_var)
            if transpose:
                tf_value = tf_value.transpose()

            if var == "shared/shared/weight:0":
                emb = self.shared
                model.config.vocab_size = emb.vocab_size = tf_value.shape[0]
                emb.build(None)
            self.weights[idx].assign(tf_value)
            done_idxs.add(idx)

            if tf_value.shape != self.weights[idx].shape:
                logger.error("Mismatch in shape: TF variable %s, TF shape %s, transformers shape %s", tf_var, tf_value.shape, self.weights[idx].shape)
                errors = True
                continue

        for idx, weight_name in enumerate(names):
            if idx not in done_idxs:
                logger.warning("Weight %s not loaded", weight_name)

        if errors:
            logger.error("There were errors during conversion")
            raise ValueError()


def fix_hdf5(fn):
    logger.info("Fixing HDF5 file: %s (shared weights)", fn)
    with h5py.File(fn, 'r+') as h5:
        shared = h5["shared"]
        shared.attrs["weight_names"] = np.array([b"t5_training/shared/weight:0"])
        grp1 = shared.create_group("t5_training")
        grp2 = grp1.create_group("shared")
        grp2["weight:0"] = shared["weight:0"]
        del shared["weight:0"]
            
        
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-10s %(message)s', level=logging.DEBUG)

    args = parser.parse_args()

    logger.info("Loading base transformer %s", args.base)
    model = T5Training.from_pretrained(args.base)
    logger.info("Loading TF checkpoint %s", args.checkpoint)
    load_checkpoint(model, args.checkpoint, assign=True)
    logger.info("Saving into %s", args.output)
    model.save_pretrained(args.output)
    fix_hdf5(os.path.join(args.output, "tf_model.h5"))
