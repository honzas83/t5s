# t5s - T5 made simple

T5 is a shortcut for the _Text-to-text transfer transformer_ - the pre-trained self-attention network developed by Google (https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html).

## Instalation

Install via Python PIP directly from GitHub:

```shell
pip install git+https://github.com/honzas83/t5s
```

This will install the t5s library together with it's dependencies: tensorflow, Huggingface Transformers, sklearn etc.


## Usage

### Prepare TSV datasets

Prepare simple tab-separated (TSV) datasets containing two columns: `input text` `\t` `output text`. Split the data into training, development and test data.

### Write a simple YAML configuration

The YAML specifies the name of pre-trained T5 model, the SentencePiece model, the names of TSV files and parameters for fine-tuning such as number of epochs, batch sizes and learning rates.

```yaml
cat > aclimdb.yaml <<EOF
tokenizer:
    spm: 'cc_all.32000/sentencepiece.model'

t5_model:
    pre_trained: "t5-base"
    save_checkpoint: "T5_aclImdb"
    save_checkpoint_every: 1

dataset:
    train_tsv: "aclImdb.train.tsv"
    devel_tsv: "aclImdb.dev.tsv"
    test_tsv: "aclImdb.test.tsv"

    loader:
        input_size: 3072
        output_size: 256
        min_batch_size: 4
        shuffle_window: 10000

training:
    shared_trainable: False
    encoder_trainable: True

    n_epochs: 20
    initial_epoch: 0

    steps_per_epoch: 10000

    learning_rate: 0.001
    learning_rate_schedule: True

predict:
  batch_size: 50
  max_input_length: 768
  max_output_length: 64
EOF
```

### Fine-tune

Use the script `t5_fine_tune.py` to fine tune the pre-trained T5 model:

```shell
t5_fine_tune.py aclimdb.yaml
```

The fine-tuning script trains the T5 model and saves it's parameters into a model directory `T5_aclImdb`. It also stores the training settings in a new YAML file `T5_aclImdb.yaml` and this configuration could be used in subsequent call to the prediction script.

### Predict

The `t5_predict.py` is used to predict using the fine-tuned T5 model, the input could be TSV or a simple TXT file:

```shell
t5_predict.py T5_aclImdb.yaml aclImdb.test.tsv aclImdb.test.pred.tsv
```

### Evaluate

The t5s library comes also with a simple script for evaluation of reference and hypothesised TSV files:

```shell
eval_tsv.py match aclImdb.dev.tsv aclImdb.dev.pred.tsv
```

## Use from Python code

The whole library could be used directly from Python code, look inside the [examples](examples) directory. You can start with the [ACL IMDB sentiment analysis dataset](examples/t5s_aclimdb.ipynb).

### Generate hidden state outputs

The Hugginface Transformers library does not allow to generate the hidden states together in the `generate()` method. The `t5s` library includes the patched version of `_generate_no_beam_search()` method which accumulates the hidden states during generation of the output.

```python
import t5s

t5 = t5s.T5(config)
batch = ["input1", "input2"]
output, hidden_states = model.predict(batch, generate_hidden_states=True)
```

The variable `hidden_states` is a list of tensors with shape `(len(batch), decoded_length, hidden_size)`. The first item in the list is the output of the output embedding, the outputs of 12 hidden layers follows and the last item is the output logits before applying the output softmax function. In other words for the `t5-base` architecture the list has 14 items and the `hidden_size` id 768 except the last logits output which has the same size as the SentencePiece vocabulary.
