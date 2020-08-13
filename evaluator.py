import os
import sys
import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import json
import pandas as pd
from functions import tokenize


class Evaluator:
    def __init__(self, model_path, tokenizer_path, param_path, testset_path):
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.max_len, self.pad_type, self.trunc_type = self.load_parameters(param_path)
        self.sentences, self.targets = self.load_testset(testset_path)

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)

    def load_tokenizer(self, tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            return pickle.load(f)

    def load_parameters(self, param_path):
        with open(param_path, "r") as f:
            params = json.load(f)

        return params["max_len"], params["pad_type"], params["trunc_type"]

    def load_testset(self, testset_path):
        df = pd.read_csv(
            testset_path, sep="\t", header=None, names=["sentence", "label"]
        )

        sentences = tokenize(df["sentence"])
        targets = np.array(df["label"].tolist())
        return sentences, targets

    def get_seqeunces(self):
        sequences = self.tokenizer.texts_to_sequences(self.sentences)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding=self.pad_type,
            truncating=self.trunc_type,
        )

        return padded

    def predict(self):
        padded = self.get_seqeunces()
        result = self.model.predict(padded)
        predicted = result > 0.5
        accuracy = np.mean(tf.squeeze(predicted) == self.targets)
        return accuracy


def parse_args(argv):  # save parsed arguments into args!
    """Parse command line arguments.
    """
    # generate parser
    parser = argparse.ArgumentParser(description=__doc__)
    # set the argument fomats

    parser.add_argument(
        "--model", "-m", default=os.path.join(".", "./LSTM/ckpt-loss=0.35"),
    )
    parser.add_argument(
        "--tokenizer", "-t", default=os.path.join(".", "tokenizer.pickle"),
    )
    parser.add_argument(
        "--param", "-p", default=os.path.join(".", "parameter.json"),
    )
    parser.add_argument(
        "--dataset", "-d", default=os.path.join(".", "review.tsv"),
    )

    return parser.parse_args(argv[1:])


if __name__ == "__main__":

    args = parse_args(sys.argv)
    evaluator = Evaluator(
        args.model, args.tokenizer, args.param, args.dataset
    )
    print(evaluator.predict())
