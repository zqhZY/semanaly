# -*- coding:utf-8 -*-
from gensim.models import Word2Vec, Doc2Vec
import sys
import jieba
reload(sys)
sys.setdefaultencoding("utf-8")


class ResultInfo(object):
    def __init__(self, index, score, score_d2v, text):
        self.id = index
        self.score = score
        self.score_d2v = score_d2v
        self.text = text


target = "./dataset/target.txt"
model = "./model/w2v.mod"
model_d = './model/model_new.doc2vec'

model_w2v = Word2Vec.load(model)
model_d2v = Doc2Vec.load(model_d)

candidates = []
with open(target) as f:
    for line in f:
        candidates.append(line.decode("utf-8").strip().split())

# print model_w2v.n_similarity(candidates[1], candidates[1])

while True:
    text = raw_input("input sentence: ").decode("utf-8")
    words = list(jieba.cut(text.strip(), cut_all=False))
    flag = False
    for w in words:
        if w not in model_w2v.wv.vocab:
            print "input word %s not in dict. skip this turn" % w
            flag = True
    if flag:
        continue

    res = []
    index = 0
    for candidate in candidates:
        # print candidate
        for c in candidate:
            if c not in model_w2v.wv.vocab:
                print "candidate word %s not in dict. skip this turn" % c
                flag = True
        if flag:
            break
        score = model_w2v.n_similarity(words, candidate)
        score_d2v = model_d2v.docvecs.similarity_unseen_docs(model_d2v, words, candidate)
        res.append(ResultInfo(index, score, score_d2v, " ".join(candidate)))
        index += 1
    res.sort(cmp=None, key=lambda x:x.score, reverse=True)
    k = 0
    for i in res:
        k += 1
        print "text %s: %s, score : %s, doc2vec score : %s " % (i.id, i.text, i.score, i.score_d2v)
        if k > 9:
            break
