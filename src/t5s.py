from tensorflow_text.python.ops.sentencepiece_tokenizer import SentencepieceTokenizer as TFSentencepieceTokenizer
import tensorflow as tf
from transformers import (T5Tokenizer, 
                          TFT5ForConditionalGeneration)
from tensorflow.keras.callbacks import LearningRateScheduler, Callback

import logging
import yaml

def sparse_from_dense(t):
    idx = tf.where(tf.not_equal(t, 0))
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
        text = parts[0]
        label = parts[1]
        return (text, label)

    def filter_labels(text, label):
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



class T5(object):
    logger = logging.getLogger("t5_lib.T5")

    def __init__(self, config):
        """
        """
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as fr:
                self.config = yaml.safe_load(fr)
        else:
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
            return

        # Load the pre-trained model
        model_config = self.config["t5_model"]
        if "load_checkpoint" in model_config:
            model_fn = model_config["load_checkpoint"]
        else:
            model_fn = model_config["pre_trained"]

        self.logger.info("Loading model from %s", model_fn)
        self.model = T5Training.from_pretrained(model_fn)

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

