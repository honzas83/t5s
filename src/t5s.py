from tensorflow_text.python.ops.sentencepiece_tokenizer import SentencepieceTokenizer as TFSentencepieceTokenizer
import tensorflow as tf
from transformers import (T5Tokenizer, 
                          TFT5ForConditionalGeneration)
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, EarlyStopping
from sklearn.metrics import precision_recall_fscore_support
import logging
import yaml
import numpy as np
import sys
import os


def remove_last_ext(fn):
    "Returns the filename with the last extension removed"
    return fn.rsplit(".", 1)[0]

# SentencePiece ids as required in the T5 trainig code

PAD_ID = 0
EOS_ID = 1
UNK_ID = 2

def sparse_from_dense(t):
    """Helper function for edit_accuracy()

    Args:
        t: Tensor of type tf.int32
        Returns:
            SparseTensor without padding and eos tokens
    """
    idx = tf.where(tf.logical_and(tf.not_equal(t, PAD_ID), tf.not_equal(t, EOS_ID)))
    shape = tf.shape(t)
    shape = tf.cast(shape, dtype=tf.int64)
    return tf.SparseTensor(idx, tf.gather_nd(t, idx), shape)


def edit_accuracy(y_true, y_pred):
    y_true = sparse_from_dense(y_true)
    y_pred = sparse_from_dense(y_pred)
    
    dist = tf.edit_distance(y_true, y_pred)
    
    acc = tf.map_fn(lambda d: tf.cond(tf.math.is_finite(d), lambda: 1-d, lambda: 0.),
                    dist)
    
    return acc


class EditAccuracy(tf.python.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='edit_accuracy', dtype=None):
        super(EditAccuracy, self).__init__(edit_accuracy, name, dtype=dtype)


def sent_accuracy(y_true, y_pred, mask=None):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    if mask is None:
        mask = tf.cast(y_true != 0, tf.int32)
        y_pred = y_pred * mask

    equal = tf.cast(y_true == y_pred, tf.int32)

    mul = tf.math.reduce_prod(equal, axis=-1)
    ref = tf.ones_like(mul)

    return mul


class SentAccuracy(tf.python.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='sent_accuracy', dtype=None):
        super(SentAccuracy, self).__init__(sent_accuracy, name, dtype=dtype)


class T5Training(TFT5ForConditionalGeneration):
    # https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-%20Training.ipynb

    def __init__(self, *args, log_dir=None, cache_dir= None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker= tf.keras.metrics.Mean(name='loss') 

    @tf.function
    def train_step(self, data):
        x, _ = data
        y = x["labels"]
        mask = x["decoder_attention_mask"]
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)
            
            grads = tape.gradient(loss, self.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)        
        self.compiled_metrics.update_state(y, tf.math.argmax(logits, axis=-1, output_type=tf.int32))
        metrics = {m.name: m.result() for m in self.metrics}
        
        return metrics

    def test_step(self, data):
        x, _ = data
        y = x["labels"]
        mask = x["decoder_attention_mask"]
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]
        
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, tf.math.argmax(logits, axis=-1, output_type=tf.int32))
        return {m.name: m.result() for m in self.metrics}


def tsv_dataset(fn, tf_tokenizer, input_size=1024, output_size=1280, min_batch_size=2,
                shuffle_window=None, line_counter=None, skip=None, repeat=False):
    """Creates TF dataset from TSV file

    The dataset uses variable-length and variable-sized batches not exceeding
    input_size tokens.  Each batch has at least min_batch_size samples, i.e.
    the maximum sequence length is limited to input_size / min_batch_size.

    The dataset is optionaly shuffled if shuffle_window is a positive integer.
    """
    input_size = tf.constant(input_size, tf.int32)
    output_size = tf.constant(output_size, tf.int32)
    min_batch_size = tf.constant(min_batch_size, tf.int64)

    if line_counter is not None:
        line_counter.assign(0 if skip is None else skip)

    def split_line(line):
        if line_counter is not None:
            line_counter.assign_add(1)
        parts = tf.strings.split(line, "\t", 1)
        parts = tf.cond(tf.shape(parts)[0] == 2, lambda: parts, lambda: tf.stack([parts[0], tf.constant("")]))
        text = parts[0]
        label = parts[1]
        return (text, label)

    def filter_labels(text, label):
        # TODO: add counter of ignored examples
        return tf.strings.length(label) > 0
        
    def tokenize(text, label):
        text = tf_tokenizer.tokenize(text)
        text_att = tf.cast(tf.math.not_equal(text, 0), tf.int32)
        
        label = tf_tokenizer.tokenize(label)
        label_att = tf.cast(tf.math.not_equal(label, 0), tf.int32)
        return text, text_att, label, label_att

    def to_dict(text, text_att, label, label_att):
        batch_size = tf.shape(text)[0]
        
        input_len = input_size // batch_size
        output_len = output_size // batch_size
        
        return ({
            "input_ids": text[:, :input_len],
            "attention_mask": text_att[:, :input_len],
            "labels": label[:, :output_len],
            "decoder_attention_mask": label_att[:, :output_len],
        }, None)

    def key_func(text, text_att, label, label_att):
        in_len = tf.cast(tf.shape(text)[0], tf.int64) 
        in_per_batch = tf.cast(input_size, tf.int64) // in_len

        out_len = tf.cast(tf.shape(label)[0], tf.int64) 
        out_per_batch = tf.cast(output_size, tf.int64) // out_len
        
        return tf.maximum(min_batch_size, tf.minimum(in_per_batch, out_per_batch))

    def reduce_func(key, dataset):
        return dataset.padded_batch(key)
        
    def window_size_func(key):
        return key

    if isinstance(fn, list):
        dataset = tf.data.TextLineDataset(fn, num_parallel_reads=len(fn))
    else:
        dataset = tf.data.TextLineDataset(fn)
    if repeat:
        dataset = dataset.repeat()
    if skip:
        dataset = dataset.skip(skip)
    dataset = (dataset.map(split_line)
                      .filter(filter_labels)
                      .map(tokenize))

    if shuffle_window is not None:
        dataset = dataset.shuffle(shuffle_window, reshuffle_each_iteration=True)

    dataset = (dataset.apply(tf.data.experimental.group_by_window(
                                  key_func=key_func,
                                  reduce_func=reduce_func,
                                  window_size_func=window_size_func
                            ))
                      .map(to_dict)
                      .prefetch(tf.data.experimental.AUTOTUNE)
                )
    return dataset


def SqrtScheduler(learning_rate, verbose=0):
    return LearningRateScheduler(lambda n: learning_rate*1/((n+1)**0.5), verbose=verbose)


class CheckpointSaver(Callback):
    logger = logging.getLogger("t5s.CheckpointSaver")

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
        self.logger.info("Consumed %d training examples", self.line_counter.value().numpy())
        if self.freq is not None and epoch % self.freq == 0:
            self.save()

    def save(self):
        if self.last_saved_epoch == self.epoch:
            # skip save on_train_end, it was already done on_epoch_end
            return

        out_fn = self.config["t5_model"]["save_checkpoint"]
        self.logger.info("Saving checkpoint to %s", out_fn)
        self.model.save_pretrained(out_fn)
        self.config["t5_model"]["load_checkpoint"] = self.config["t5_model"]["save_checkpoint"]
        if self.epoch is not None:
            self.config["training"]["initial_epoch"] = self.epoch+1
            if "steps_per_epoch" in self.config["training"]:
                self.config["training"]["skip_samples"] = self.line_counter.value().numpy().item()

        out_yaml_fn = self.config["t5_model"]["save_checkpoint"]+".yaml"
        with open(out_yaml_fn, "w", encoding="utf-8") as fw:
            yaml.dump(self.config, fw, default_flow_style=False, sort_keys=True)

        self.last_saved_epoch = self.epoch


class T5(object):
    logger = logging.getLogger("t5s.T5")

    def __init__(self, config):
        """
        """
        if isinstance(config, str):
            self.config_fn = config
            with open(config, "r", encoding="utf-8") as fr:
                self.config = yaml.safe_load(fr)
        else:
            self.config_fn = None
            self.config = config
        self.model = None
        self.predict_tokenizers = None

    def load_tokenizer(self, type="predict"):
        assert type == "predict"

        self.logger.info("Loaded tokenizer from: %s", self.config["tokenizer"]["spm"])
        tokenizer = T5Tokenizer(self.config["tokenizer"]["spm"])
        with open(self.config["tokenizer"]["spm"], "rb") as f:
            tf_tokenizer = TFSentencepieceTokenizer(f.read(), add_eos=True)
        return tokenizer, tf_tokenizer

    def load_model(self):
        if self.model is not None:
            return self.model

        # Load the pre-trained model
        model_config = self.config["t5_model"]
        if "load_checkpoint" in model_config:
            model_fn = model_config["load_checkpoint"]
        else:
            model_fn = model_config["pre_trained"]

        self.logger.info("Loading model from %s", model_fn)
        self.model = T5Training.from_pretrained(model_fn)
        return self.model

    def predict(self, batch):
        if self.predict_tokenizers is None:
            self.predict_tokenizers = self.load_tokenizer("predict")

        self.load_model()

        predict_config = self.config.get("predict", {})

        max_input_length = predict_config.get("max_input_length", None)
        min_output_length = predict_config.get("min_output_length", None)
        max_output_length = predict_config.get("max_output_length", None)
        no_repeat_ngram_size = predict_config.get("no_repeat_ngram_size", 0)
        length_penalty = predict_config.get("length_penalty", 1.0)

        tokenizer, tf_tokenizer = self.predict_tokenizers

        sentences = tokenizer(batch, padding="longest", max_length=max_input_length, truncation=True)
        input_ids = tf.constant(sentences["input_ids"])
        outputs = self.model.generate(input_ids, 
                                      min_length=min_output_length,
                                      max_length=max_output_length, 
                                      early_stopping=True,
                                      no_repeat_ngram_size=no_repeat_ngram_size,
                                      length_penalty=length_penalty)

        preds = tf_tokenizer.detokenize(outputs).numpy()
        preds = [i.decode() for i in preds]
        return preds
    
    def predict_tsv(self, tsv_in, tsv_out):
        batch_size = self.config.get("predict", {}).get("batch_size", 400)

        with open(tsv_in, "r", encoding="utf-8") as fr, \
             open(tsv_out, "w", encoding="utf-8") as fw:

            batch = []

            def flush(n_predicted=[0]):
                preds = self.predict(batch)

                for input_sent, output_sent in zip(batch, preds):
                    n_predicted[0] += 1
                    print(input_sent, output_sent, sep="\t", file=fw)
                    fw.flush()
                    
                del batch[:]
                self.logger.info("Processed %d items", n_predicted[0])

            for line in fr:
                items = line.strip().split("\t")
                input_sent = items[0]
                batch.append(input_sent)
                if len(batch) >= batch_size:
                    flush()
            else:
                if batch:
                    flush()

    def fine_tune(self):
        # Initialize configuration variables
        dataset_config = self.config.get("dataset", {})
        training_config = self.config.get("training", {})

        train_tsv = dataset_config["train_tsv"]
        devel_tsv = dataset_config["devel_tsv"]

        dataset_kwargs = dataset_config.get("loader", {})

        steps_per_epoch = training_config.get("steps_per_epoch", None)
        skip_samples = training_config.get("skip_samples", None)

        learning_rate = training_config.get("learning_rate", 1e-4)
        learning_rate_schedule = training_config.get("learning_rate_schedule", True)

        early_stopping = training_config.get("early_stopping", False)
        if isinstance(early_stopping, dict):
            # We have the configuration section for early stopping
            # enable early_stopping and use the dict with details
            early_stopping, early_stopping_config = True, early_stopping
        else:
            # No detailed configuration for early stopping, use empty dict
            early_stopping_config = {}

        # Load the SentencePiece tokenizer
        tokenizer, tf_tokenizer = self.load_tokenizer()

        model = self.load_model()

        # Configure trainable variables
        model.shared.trainable = training_config["shared_trainable"]
        model.encoder.trainable = training_config["encoder_trainable"]

        # Initialize metrics
        metrics = [SentAccuracy(), EditAccuracy()]

        # Initialize optimizer
        optimizer = "adam"

        model.compile(optimizer=optimizer, metrics=metrics)

        callbacks = []
        if learning_rate_schedule:
            callbacks.append(SqrtScheduler(learning_rate, verbose=1))

        if early_stopping:
            # Configure early stopping
            self.logger.info("Using early stopping config: %s", early_stopping_config)

            early_stopping_quantity = early_stopping_config.get("quantity", "val_loss")
            early_stopping_patience = early_stopping_config.get("patience", 0)

            callbacks.append(EarlyStopping(monitor=early_stopping_quantity,
                                           restore_best_weights=True,
                                           verbose=True,
                                           patience=early_stopping_patience))

        # Automatically generate t5_model.save_checkpoint
        if "save_checkpoint" not in self.config["t5_model"]:
            if self.config_fn is not None and self.config_fn.endswith(".init.yaml"):
                save_checkpoint = remove_last_ext(self.config_fn)
                save_checkpoint = remove_last_ext(save_checkpoint)
                self.config["t5_model"]["save_checkpoint"] = save_checkpoint
            else:
                raise ValueError("Cannot determine the value of missing t5_model.save_checkpoint")

        self.logger.info("Trained model will be saved into %s", self.config["t5_model"]["save_checkpoint"])
        checkpoint_saver = CheckpointSaver(model, self.config)
        callbacks.append(checkpoint_saver)

        # Instantiate datasets
        self.logger.debug("Dataset loader parameters: %s", dataset_kwargs)
        self.logger.info("Training dataset: %s", train_tsv)
        train_dataset_kwargs = dataset_kwargs.copy()
        if steps_per_epoch:
            train_dataset_kwargs["repeat"] = True
        if skip_samples:
            self.logger.info("Skipping initial %d samples, training starts from epoch %d",
                        skip_samples, training_config["initial_epoch"]+1)
            train_dataset_kwargs["skip"] = skip_samples
        train_dataset = tsv_dataset(train_tsv, tf_tokenizer,
                                    line_counter=checkpoint_saver.line_counter,
                                    **train_dataset_kwargs)

        self.logger.info("Development dataset: %s", devel_tsv)
        dev_dataset_kwargs = dataset_kwargs.copy()
        dev_dataset_kwargs.pop("repeat", None)
        dev_dataset_kwargs.pop("shuffle_window", None)
        dev_dataset = tsv_dataset(devel_tsv, tf_tokenizer, **dev_dataset_kwargs)

        self.model.fit(train_dataset, 
                       validation_data=dev_dataset,
                       steps_per_epoch=steps_per_epoch,
                       callbacks=callbacks,
                       initial_epoch=training_config["initial_epoch"],
                       epochs=training_config["n_epochs"])

        if "evaluation" in self.config:
            self.evaluate()

    def predict_dataset(self, dataset):
        tsv = self.config["dataset"].get("{}_tsv".format(dataset))
        if tsv is None:
            raise ValueError("No such dataset: {}".format(dataset))

        if not isinstance(tsv, list):
            tsv = [tsv]

        model_base = os.path.split(self.config["t5_model"]["save_checkpoint"])[-1]

        ref_fns = []
        hyp_fns = []
        for ref_fn in tsv:
            hyp_fn = "{ref_base}.{model_base}.tsv".format(
                        ref_base=remove_last_ext(ref_fn),
                        model_base=model_base,
                    )
            self.logger.info("Predicting %s into %s", ref_fn, hyp_fn)
            self.predict_tsv(ref_fn, hyp_fn)
            ref_fns.append(ref_fn)
            hyp_fns.append(hyp_fn)
        return ref_fns, hyp_fns

    def evaluate(self, datasets=None):
        """Executes the evaluation of the model

        The evaluation is performed for each dataset under the "dataset"
        section, with the exception of train dataset. The dataset's key must
        end with "_tsv" suffix and the name of dataset is without this suffix.

        The result is stored in YAML file with the following filename:
        "{model_base}.eval.{dataset}.yaml", where "model_base" is the path to
        model checkpoint (see "save_checkpoint" in configuration YAML) and
        "dataset" is the name of the dataset.

        The evaluation datasets could be limited with the configuration key
        "evaluation.datasets".

        Args:
            datasets: an override for "evaluation.datasets" configuration key
        """
        evaluation_cfg = self.config["evaluation"]
        metric_name = evaluation_cfg["metric"]
        metric = EVAL_METRICS[metric_name]

        if datasets is None:
            default_eval_datasets = [i[:-4] for i in self.config["dataset"] if i.endswith("_tsv") and i != "train_tsv"]
            datasets = evaluation_cfg.get("datasets", default_eval_datasets)


        for dataset in datasets:
            ref_fns, hyp_fns = self.predict_dataset(dataset)

            eval_results = eval_tsv(metric, ref_fns, hyp_fns)

            eval_fn = "{model_base}.eval.{dataset}.yaml".format(
                            model_base=self.config["t5_model"]["save_checkpoint"],
                            dataset=dataset,
                      )

            self.logger.info("Evaluation results for dataset %s:", dataset)
            with open(eval_fn, "w", encoding="utf-8") as fw:
                yaml_dump_result(eval_results, sys.stdout)
                yaml_dump_result(eval_results, fw)


#  Evaluation metrics definition


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
    return {"P": float(P), "R": float(R), "F": float(F)}


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


EVAL_METRICS = {
    "f1_multilabel": f1_multilabel,
    "match": match,
    "binary_lab": binary_lab,
}

TOTAL_METRIC = "__total__"

def eval_tsv(metric, ref, hyp):
    """Evaluates the prediction results using reference and (optionally) multiple hypothesis

    Args:
        metric: Metric function to evaluate, choose from EVAL_METRICS dictionary
        ref: Reference TSV
        hyp: Single string or list of strings, hypothesis TSV
    """
    logger = logging.getLogger("t5s.eval_tsv")

    if not isinstance(ref, list):
        ref = [ref]
    if not isinstance(hyp, list):
        hyp = [hyp]

    assert len(ref) == len(hyp)

    ret = {}
    all_pairs = []
    for ref_fn, hyp_fn in zip(ref, hyp):
        pairs = []
        with open(ref_fn, "r", encoding="utf-8") as fr_ref, \
             open(hyp_fn, "r", encoding="utf-8") as fr_hyp:
            for idx, (ref_line, hyp_line) in enumerate(zip(fr_ref, fr_hyp)):
                ref_in, ref_out = ref_line.split("\t")[:2]
                hyp_in, hyp_out = hyp_line.split("\t")[:2]
                if ref_in != hyp_in:
                    logger.warning("Reference and hypothesis inputs mismatch on line: %d", idx)

                pairs.append((ref_out, hyp_out))

            logger.info("Loaded %d examples", len(pairs))

        all_pairs.extend(pairs)
        if len(hyp) != 1:
            # Store partial results only if we have multiple files
            ret[hyp_fn] = metric(pairs)
    # Compute total metric value
    ret[TOTAL_METRIC] = metric(all_pairs)
    return ret


def yaml_dump_result(obj, stream):
    """Redefinition of yaml.safe_dump with added float representer

    The float representer uses float precision of four decimal digits
    """
    def float_representer(dumper, value):
        text = '{0:.4f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)

    class ResultDumper(yaml.SafeDumper):
        def __init__(self, *args, **kwargs):
            super(ResultDumper, self).__init__(*args, **kwargs)
            self.add_representer(float, float_representer)

    yaml.dump(obj, stream, Dumper=ResultDumper, default_flow_style=False, sort_keys=True)
