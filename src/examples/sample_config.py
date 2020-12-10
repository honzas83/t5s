config = {
    "tokenizer": {
        "spm": "cc_all.32000/sentencepiece.model",
    },
    "t5_model": {
        "pre_trained": "t5-base",
        "save_checkpoint": "T5_aclImdb",
        "save_checkpoint_every": 1,
    },
    "dataset": {
        "train_tsv": "aclImdb.train.tsv",
        "devel_tsv": "aclImdb.dev.tsv",
        "test_tsv": "aclImdb.test.tsv",
        "loader": {
            "input_size": 3072,
            "output_size": 256,
            "min_batch_size": 4,
        },
    },
    "training": {
        "shared_trainable": False,
        "encoder_trainable": True,
        "n_epochs": 20,
        "initial_epoch": 0,
        "steps_per_epoch": 1000,
        "learning_rate": 0.001,
        "learning_rate_schedule": True,
    },
}
