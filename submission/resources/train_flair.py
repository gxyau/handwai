#!/usr/bin/env python
from flair.data       import Corpus, Sentence, Token
from flair.datasets   import ColumnCorpus
from flair.embeddings import BertEmbeddings, CharacterEmbeddings, FlairEmbeddings, WordEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models     import SequenceTagger
from flair.trainers   import ModelTrainer
import json
import random
#random.seed(1234)

def train_model(train="model/train.txt", dev="model/dev.txt", test="model/test.txt"):
    # 0. Info
    columns   = {0: "text", 1: "ner"}
    data_path = "model/"
    with open("model/weight.json", 'r') as f:
        weights = json.load(f)
    
    # 1. Get the corpus
    corpus: Corpus = ColumnCorpus(data_path, columns, train_file="train.txt", dev_file="dev.txt", test_file="test.txt")

    # 2. what tag do we want to predict?
    tag = "ner"
    
    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag)

    # 4. initialize embeddings
    emb_list = [
                # Word embedding features
                WordEmbeddings("en-twitter"),
                WordEmbeddings("glove"),
                # Flair embedding features
                FlairEmbeddings("mix-forward"),
                FlairEmbeddings("mix-backward"),
                # Other embeddings
                #TransformerWordEmbeddings('bert-base-uncased')
            ]
    
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=emb_list)

    # 5. initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size    = 256,
                                            embeddings     = embeddings,
                                            tag_dictionary = tag_dictionary,
                                            tag_type       = tag,
                                            # dropout        = 0.1836444885743146,
                                            loss_weights   = weights,
                                            use_crf        = True)

    # 6. initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('output/test_model',
                  learning_rate   = 0.1,
                  mini_batch_size = 32,
                  max_epochs      = 150)

if __name__ == "__main__":
    train_f = input("Please enter train data: ").strip()
    dev_f   = input("Please enter dev data: ").strip()
    test_f  = input("Please enter test data: ").strip()
    if train_f == "" or dev_f == "" or test_f == "":
        print("Missing data paths, using default paths")
        train_model()
    else:
        train_model(train_f, dev_f, test_f)

