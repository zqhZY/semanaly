# -*- coding:utf-8 -*-
from gensim.models import Word2Vec
import sys
import jieba
reload(sys)
sys.setdefaultencoding("utf-8")


class ResultInfo(object):
    def __init__(self, index, score, text):
        self.id = index
        self.score = score
        self.text = text


target = "./dataset/target.txt"
model = "./model/w2v.mod"

model_loaded = Word2Vec.load(model)

candidates = []
with open(target) as f:
    for line in f:
        candidates.append(line.decode("utf-8").strip().split())

print model_loaded.n_similarity(candidates[1], candidates[1])

while True:
    text = raw_input("input sentence: ").decode("utf-8")
    words = list(jieba.cut(text.strip(), cut_all=False))
    print len(words)
    res = []
    index = 0
    for candidate in candidates:
        # print candidate
        score = model_loaded.n_similarity(words, candidate)
        res.append(ResultInfo(index, score, " ".join(candidate)))
        index += 1
    res.sort(cmp=None, key=lambda x:x.score, reverse=True)
    k = 0
    for i in res:
        k += 1
        print "text %s: %s, score : %s" % (i.id, i.text, i.score)
        if k > 9:
            break
