# -*- coding:utf-8 -*-
########################################
## import packages
########################################
import os
import csv
import codecs
import jieba
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.preprocessing.text
import sys
import cPickle
from lstm import get_model

reload(sys)
sys.setdefaultencoding('utf-8')


########################################
## set directories and parameters
########################################
DATA_DIR = '../dataset/'
EMBEDDING_FILE = '../model/w2v/w2v.mod'
DATA_FILE = DATA_DIR + 'mytest_pair.csv'
# TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

num_lstm = 175
num_dense = 100
rate_drop_lstm = 0.15
rate_drop_dense = 0.15

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = '../model/lstm/lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                                rate_drop_dense)

save = False
load_tokenizer = True
save_path = "../model/lstm"
tokenizer_name = "tokenizer.pkl"
embedding_matrix_path = "../model/lstm/embedding_matrix.npy"

########################################
## process texts in datasets
########################################
print('Processing text dataset')


def text_to_wordlist(text, remove_stopwords=False):
    # Clean the text, with the option to remove stopwords.

    # Convert words to lower case and split them
    words = list(jieba.cut(text.strip(), cut_all=False))

    # Optionally, remove stop words
    if remove_stopwords:
        pass
    # Return a list of words
    return (" ".join(words))


texts_1 = []
texts_2 = []
test_ids = []
with codecs.open(DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        test_ids.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))

# print texts_1

'''
this part is solve keras.preprocessing.text can not process unicode
start here
'''


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
    else:
        translate_table = keras.maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]


keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence
'''
end here
'''

if load_tokenizer:
    print('Load tokenizer...')
    tokenizer = cPickle.load(open(os.path.join(save_path, tokenizer_name), 'rb'))
else:
    print("Fit tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False)
    tokenizer.fit_on_texts(texts_1 + texts_2)
    if save:
        print("Save tokenizer...")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cPickle.dump(tokenizer, open(os.path.join(save_path, tokenizer_name), "wb"))

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
# print sequences_1

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data_1.shape)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')
word2vec = Word2Vec.load(EMBEDDING_FILE)

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.load(embedding_matrix_path)

model = get_model()
print(STAMP)

bst_model_path = STAMP + '.h5'

model.load_weights(bst_model_path)

predicts = model.predict([data_1, data_2], batch_size=10, verbose=1)
for i in range(len(test_ids)):
    print "t1: %s, t2: %s, score: %s" % (texts_1[i], texts_2[i], predicts[i])
