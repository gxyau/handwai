#!/usr/bin/env python

from flair.data import Corpus, Sentence, Token
from flair.datasets import ColumnCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.hyperparameter.param_selection import OptimizationValue, Parameter, SearchSpace, SequenceTaggerParamSelector
from hyperopt import hp
import json
from pathlib import Path
import random
from typing import List

random.seed(2020)

# Basic Info
columns   = {0: "text", 1: "ner"}
data_path = "model/"
with open("model/weight.json",'r') as f:
    weights = json.load(f)

# Get the Corpus
corpus: Corpus = ColumnCorpus(data_path, columns, train_file="train.txt", dev_file="dev.txt", test_file="test.txt")

# Tag
tag_type = 'ner'

# Tag dictionary
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# Embeddings for grid search
embedding_types = [
    # Word embedding features
    WordEmbeddings("en-twitter"),
    WordEmbeddings("glove"),
    # Flair embedding features
    FlairEmbeddings("mix-forward"),
    FlairEmbeddings("mix-backward"),
    # Other embeddings
    #TransformerWordEmbeddings('bert-base-uncased')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# Grid search parameters
search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embeddings])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128, 256, 512])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.1, 0.15, 0.2, 0.25, 0.5])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 24, 32])


param_selector = SequenceTaggerParamSelector(
    corpus,
    tag_type,
    Path('output/hyper_optimization'),
    max_epochs = 150,
    training_runs = 3,
    optimization_value = OptimizationValue.DEV_SCORE
)

param_selector.optimize(search_space, max_evals=100)
