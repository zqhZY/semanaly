# -*- encoding:utf-8 -*-
import sys
import codecs

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")

train_file = "/home/zqh/mygit/textNN/keras/CNN-text-classification-keras/data/360_train_cutted.tsv"
test_file = "/home/zqh/mygit/textNN/keras/CNN-text-classification-keras/data/360_test_cutted.tsv"
corpus = []
lables = []
stopwords = codecs.open('stop_words_ch.txt', 'r', encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]
print stopwords

train_corpus = []
with codecs.open(train_file, 'r', encoding='utf8') as f:
    for line in f:
        tokens = line.strip().split("\t")
        train_corpus.append(tokens[0])
        lables.append(0 if tokens[1] == "NEGATIVE" else 1)

test_corpus = []
with codecs.open(test_file, 'r', encoding='utf8') as f:
    for line in f:
        tokens = line.strip().split("\t")
        test_corpus.append(tokens[0])

corpus = train_corpus + test_corpus
negative_corpus = []
for s, l in zip(train_corpus, lables):
    if l == 0:
        negative_corpus.append(s)
print len(negative_corpus)
corpus = negative_corpus
print len(train_corpus)
print len(test_corpus)
print len(corpus)
print lables
cntVector = CountVectorizer(stop_words=stopwords, ngram_range=(3,3))
cntTf = cntVector.fit_transform(corpus)

for i in cntVector.get_feature_names():
    print i

print cntTf
