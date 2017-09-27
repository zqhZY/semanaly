# -*- coding:utf-8 -*-
import argparse
import sys

import runner_kfold
from dataset import MyDataset
import numpy as np
from gensim.models import Word2Vec
from torchtext import data

from models.TextLSTM import TextLSTM
from models.TextCNN import TextCNN

reload(sys)
sys.setdefaultencoding("utf-8")

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 128]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

TEXT = data.Field()
LABELS = data.Field(sequential=False)

dataset = MyDataset(path='./dataset/demo_dataset.csv', format='csv', fields=[('text', TEXT), ('labels', LABELS)])

LABELS.build_vocab(dataset.labels)
TEXT.build_vocab(dataset.text)
print(TEXT.vocab.freqs.most_common(10))
print(LABELS.vocab.itos)
# print(train.fields)
# print(len(train))
# print(vars(train[0]))


EMBEDDING_FILE = '../model/w2v.mod'
MAX_NB_WORDS = 50000
embedding_matrix_path = "./embedding_matrix.npy"
########################################
# prepare embeddings
########################################

print('Preparing embedding matrix')
word2vec = Word2Vec.load(EMBEDDING_FILE)

nb_words = min(MAX_NB_WORDS, len(TEXT.vocab))

embedding_matrix = np.zeros((nb_words, 100))
c = 0
for i, word in enumerate(TEXT.vocab.itos):
    if word in word2vec.wv.vocab:
        embedding_matrix[i] = word2vec.wv.word_vec(word)
    else:
        c += 1
        # print word
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
np.save(embedding_matrix_path, embedding_matrix)


args.vocab_size = len(TEXT.vocab)
args.hidden_size = 100
args.linear_hidden_size = 100
args.num_classes = 3
args.embedding_path = embedding_matrix_path
args.kernel_sizes = [3, 4, 5]
args.save_dir = "./snapshots"
args.cuda = False

kfold_range = dataset.kfold(5)
accuracys = []
avg_losses = []

for train_range, test_range in kfold_range:
    print type(train_range)
    train, dev = dataset.get_fold(fields=[('text', TEXT), ('labels', LABELS)], train_indexs=train_range, test_indexs=test_range)
    train_iter, dev_iter = data.Iterator.splits((train, dev), device=-1, batch_sizes=(args.batch_size, len(dev)))

    cnn = TextCNN(args)
    cnn = runner_kfold.fit(train_iter, dev_iter, cnn, args)

    accuracy, loss = runner_kfold.eval(dev_iter, cnn)
    accuracys.append(accuracy)
    avg_losses.append(loss)

print("avarage accuracy is %s, loss is %s".format(np.average(accuracys)), np.average(avg_losses))

