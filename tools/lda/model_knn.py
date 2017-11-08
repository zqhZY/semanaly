# -*- encoding:utf-8 -*-
import random
import sys
import codecs
from sklearn import svm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from result_show import plot_confusion_matrix

reload(sys)
sys.setdefaultencoding("utf-8")

train_file = "/home/zqh/mygit/cincc/semanaly/tools/data/mobile_dataset_jieba.csv"
test_file = "/home/zqh/mygit/cincc/semanaly/tools/data/mobile_dataset_jieba_test_cleaned.csv"
lables = []
stopwords = codecs.open('stop_words_ch.txt', 'r', encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]
# print stopwords

label_map = {u"办理":0, u"投诉（含抱怨）":1, u"咨询（含查询）":2, u"其他":3, u"表扬及建议":4}
train_corpus = []
with codecs.open(train_file, 'r', encoding='utf8') as f:
    first_line = True
    for line in f:
        if first_line:
            first_line = False
            continue
        tokens = line.strip().split(",")
        train_corpus.append([tokens[1], label_map[tokens[2]]])
        # print tokens[2]
        # print line
        # lables.append(label_map[tokens[2]])

test_corpus = []
ans = []
with codecs.open(test_file, 'r', encoding='utf8') as f:
    first_line = True
    for line in f:
        if first_line:
            first_line = False
            continue
        tokens = line.strip().split(",")
        test_corpus.append(tokens[1])
        ans.append(label_map[tokens[2]])

random.shuffle(train_corpus)
train_text = []
lables = []
for t, l in train_corpus:
    train_text.append(t)
    lables.append(l)

print len(train_text)
print len(test_corpus)

count_vector = CountVectorizer(stop_words=stopwords)
wordcount_train = count_vector.fit_transform(train_text)
wordcount_test = count_vector.transform(test_corpus)

# tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
#                              stop_words=stopwords)
# X_train = tfidf_vectorizer.fit_transform(train_text)
# X_test = tfidf_vectorizer.transform(test_corpus)
select_chi2 = True
select_chi2 = 23060
if select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          select_chi2)
    ch2 = SelectKBest(chi2, k=select_chi2)
    X_train = ch2.fit_transform(wordcount_train, lables)
    X_test = ch2.transform(wordcount_test)
    print()



classifier = MultinomialNB(alpha=0.001)
classifier.fit(X_train, lables)

preds = classifier.predict(X_test)

class_names = [u"办理", u"投诉（含抱怨）", u"咨询（含查询）", u"其他", u"表扬及建议"]
print "accuracy: ", accuracy_score(ans, preds)
print "F1: ", f1_score(ans, preds, average=None)
cnf_matrix = confusion_matrix(ans, preds)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()

