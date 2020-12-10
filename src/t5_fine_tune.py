#!/usr/bin/env python3
# coding: utf-8


import os
import argparse
import logging
import yaml
from t5s import (T5Tokenizer,
                 T5Training,
                 TFSentencepieceTokenizer,
                 tsv_dataset, SentAccuracy, EditAccuracy,
                 SqrtScheduler, Callback, tf)


logger = logging.getLogger('t5s.fine_tune')

parser = argparse.ArgumentParser(description='T5 fine-tuner')
parser.add_argument('config', metavar='YAML', type=str, help='Configuration YAML file')


class CheckpointSaver(Callback):
    def __init__(self, model, config):
        super(CheckpointSaver, self).__init__()
        self.model = model
        self.config = config
        self.freq = config["t5_model"].get("save_checkpoint_every", None)
        self.epoch = None
        self.last_saved_epoch = None
        self.line_counter = tf.Variable(0, trainable=False, name="line_counter")

    def on_train_end(self, logs=None):
        self.save()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        print()
        logger.info("Consumed %d training examples", self.line_counter.value().numpy())
        if self.freq is not None and epoch % self.freq == 0:
            self.save()

    def save(self):
        if self.last_saved_epoch == self.epoch:
            # skip save on_train_end, it was already done on_epoch_end
            return

        out_fn = self.config["t5_model"]["save_checkpoint"]
        logger.info("Saving checkpoint to %s", out_fn)
        model.save_pretrained(out_fn)
        self.config["t5_model"]["load_checkpoint"] = self.config["t5_model"]["save_checkpoint"]
        if self.epoch is not None:
            self.config["training"]["initial_epoch"] = self.epoch+1
            if "steps_per_epoch" in self.config["training"]:
                self.config["training"]["skip_samples"] = self.line_counter.value().numpy().item()

        out_yaml_fn = self.config["t5_model"]["save_checkpoint"].rsplit(".", 1)[0]+".yaml"
        with open(out_yaml_fn, "w", encoding="utf-8") as fw:
            yaml.dump(self.config, fw, default_flow_style=False)

        self.last_saved_epoch = self.epoch

        

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-10s %(message)s', level=logging.DEBUG)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fr:
        config = yaml.safe_load(fr)

    # Initialize configuration variables
    train_tsv = config["dataset"]["train_tsv"]
    devel_tsv = config["dataset"]["devel_tsv"]
    test_tsv = config["dataset"]["test_tsv"]

    dataset_kwargs = config["dataset"].get("loader", {})
    steps_per_epoch = config["training"].get("steps_per_epoch", None)
    skip_samples = config["training"].get("skip_samples", None)

    learning_rate = config["training"].get("learning_rate", 1e-4)
    learning_rate_schedule = config["training"].get("learning_rate_schedule", True)

    # Load the SentencePiece tokenizer
    logger.info("Loaded tokenizer from: %s", config["tokenizer"]["spm"])
    tokenizer = T5Tokenizer(config["tokenizer"]["spm"])
    with open(config["tokenizer"]["spm"], "rb") as f:
        bin_data = f.read()
        enable_sampling = config["tokenizer"].get("enable_sampling", False)
        alpha = config["tokenizer"].get("alpha", 1.0)
        if enable_sampling:
            logger.info("Initializing SentencepieceTokenizer with sampling enabled and alpha: %s", alpha)
            tf_train_tokenizer = TFSentencepieceTokenizer(bin_data, add_eos=True, enable_sampling=True, alpha=alpha)
            tf_dev_tokenizer = TFSentencepieceTokenizer(bin_data, add_eos=True)
        else:
            tf_train_tokenizer = TFSentencepieceTokenizer(bin_data, add_eos=True)
            tf_dev_tokenizer = tf_train_tokenizer

    # Load the pre-trained model
    model_config = config["t5_model"]
    if "load_checkpoint" in model_config:
        model_fn = model_config["load_checkpoint"]
    else:
        model_fn = model_config["pre_trained"]

    logger.info("Loading model from %s", model_fn)
    model = T5Training.from_pretrained(model_fn)

    # Configure trainable variables
    model.shared.trainable = config["training"]["shared_trainable"]
    model.encoder.trainable = config["training"]["encoder_trainable"]

    # Initialize metrics
    metrics = [SentAccuracy(), EditAccuracy()]

    # Initialize optimizer
    optimizer = "adam"

    model.compile(optimizer=optimizer, metrics=metrics)

    callbacks = []
    if learning_rate_schedule:
        callbacks.append(SqrtScheduler(learning_rate, verbose=1))

    logger.info("Trained model will be saved into %s", config["t5_model"]["save_checkpoint"])
    checkpoint_saver = CheckpointSaver(model, config)
    callbacks.append(checkpoint_saver)

    # Instantiate datasets
    logger.debug("Dataset loader parameters: %s", dataset_kwargs)
    logger.info("Training dataset: %s", train_tsv)
    train_dataset_kwargs = dataset_kwargs.copy()
    if steps_per_epoch:
        train_dataset_kwargs["repeat"] = True
    if skip_samples:
        logger.info("Skipping initial %d samples, training starts from epoch %d",
                    skip_samples, config["training"]["initial_epoch"]+1)
        train_dataset_kwargs["skip"] = skip_samples
    train_dataset = tsv_dataset(train_tsv, tf_train_tokenizer,
                                line_counter=checkpoint_saver.line_counter,
                                **train_dataset_kwargs)

    logger.info("Development dataset: %s", devel_tsv)
    dev_dataset_kwargs = dataset_kwargs.copy()
    dev_dataset_kwargs.pop("repeat", None)
    dev_dataset_kwargs.pop("shuffle_window", None)
    dev_dataset = tsv_dataset(devel_tsv, tf_dev_tokenizer, **dev_dataset_kwargs)


    model.fit(train_dataset, 
              validation_data=dev_dataset,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks,
              initial_epoch=config["training"]["initial_epoch"],
              epochs=config["training"]["n_epochs"])

