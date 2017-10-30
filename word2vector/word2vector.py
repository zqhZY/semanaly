# -*- coding:utf-8 -*-

import logging
from gensim.models.word2vec import LineSentence, Word2Vec
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

sentences= LineSentence("../dataset/train_questions_with_evidence.txt")

model = Word2Vec(sentences ,min_count=1, iter=1000)
model.train(sentences, total_examples=model.corpus_count, epochs=1000)

model.save("../model/w2v.mod")
model_loaded = Word2Vec.load("../model/w2v.mod")

sim = model_loaded.wv.most_similar(positive=[u'酒精'])
for s in sim:
    print s[0]

