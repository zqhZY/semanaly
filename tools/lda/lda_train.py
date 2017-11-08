# -*- encoding:utf-8 -*-
import sys
import codecs

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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

print len(train_corpus)
print len(test_corpus)
print len(corpus)
print lables
cntVector = CountVectorizer(stop_words=stopwords)
cntTf = cntVector.fit_transform(corpus)
print len(cntVector.get_feature_names())
np.save("count.npy", cntTf)


lda = LatentDirichletAllocation(n_topics=4,
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)

np.save("lda_docres.npy", docres)
print type(docres)

train_features = docres[0:len(train_corpus)]
test_features = docres[len(train_corpus):]

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_features, lables)

print(clf.feature_importances_)

print(clf.predict(test_features))



import pickle
pickle.dump(clf, open("rf.mod", "w"))

obj = pickle.load(open("rf.mod"))
print(obj.predict(test_features))