# -*- coding:utf-8 -*-

import gensim
import smart_open
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


data_set = "../dataset/train_questions_with_evidence.txt"

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                text = line.split()
                yield TaggedDocument(text[:], [i])


train_data = list(read_corpus(data_set))
print len(train_data)

model = Doc2Vec(size=100, min_count=1, iter=1000)
# model.build_vocab(train_data)
# # #
model.train(train_data, total_examples=model.corpus_count, epochs=1000)
# #
# # # store the model to mmap-able files
# model.save('./model/model_new.doc2vec')
# load the model back
model_loaded = Doc2Vec.load('../model/model_new.doc2vec')
#
# print model_loaded.infer_vector([u'消费者', u'消费者'])
doc_sims = model_loaded.docvecs.most_similar([2])
print doc_sims

sims = model_loaded.most_similar(positive=["酒精"], topn=10)
for s in sims:
    print s[0]

doc_id = 36
# inferred_vector = model_loaded.infer_vector(['酒精', '用', '啥', '稀释'])
inferred_vector = model_loaded.infer_vector(train_data[doc_id].words)
sims = model_loaded.docvecs.most_similar([inferred_vector], topn=len(model_loaded.docvecs))

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_data[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model_loaded)
for label, index in [('MOST', 0), ("SECOND", 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    print sims[index][1]
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_data[sims[index][0]].words)))
# # print model.docvecs[[299]]
