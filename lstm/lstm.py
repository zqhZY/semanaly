# -*- coding:utf-8 -*-
########################################
## import packages
########################################
import os
import re
import csv
import codecs

import jieba
import numpy as np

from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.preprocessing.text

import sys
import cPickle

reload(sys)
sys.setdefaultencoding('utf-8')

########################################
# set directories and parameters
########################################
DATA_DIR = '../dataset/'
EMBEDDING_FILE = '../model/w2v/w2v.mod'
TRAIN_DATA_FILE = DATA_DIR + 'mytrain_pair.csv'
TEST_DATA_FILE = DATA_DIR + 'mytest_pair.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

# num_lstm = np.random.randint(175, 275)
# num_dense = np.random.randint(100, 150)
# rate_drop_lstm = 0.15 + np.random.rand() * 0.25
# rate_drop_dense = 0.15 + np.random.rand() * 0.25

num_lstm = 175
num_dense = 100
rate_drop_lstm = 0.15
rate_drop_dense = 0.15

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = '../model/lstm/lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                                rate_drop_dense)

save = True
load_tokenizer = False
save_path = "../model/lstm"
tokenizer_name = "tokenizer.pkl"
embedding_matrix_path = "../model/lstm/embedding_matrix.npy"

# ########################################
# ## index word vectors
# ########################################
# print('Indexing word vectors')
#
# word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
# print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
# process texts in datasets
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
    return " ".join(words)


texts_1 = []
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[3]))
        test_texts_2.append(text_to_wordlist(values[4]))
        test_ids.append(values[0])
print('Found %s texts in train.csv' % len(test_texts_1))

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
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
    if save:
        print("Save tokenizer...")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cPickle.dump(tokenizer, open(os.path.join(save_path, tokenizer_name), "wb"))

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
# print sequences_1

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

########################################
# prepare embeddings
########################################
print('Preparing embedding matrix')
word2vec = Word2Vec.load(EMBEDDING_FILE)

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.wv.vocab:
        embedding_matrix[i] = word2vec.wv.word_vec(word)
    else:
        print word
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

np.save(embedding_matrix_path, embedding_matrix)


# #######################################
# # sample train/validation data
# #######################################
# np.random.seed(1234)
# perm = np.random.permutation(len(data_1))
# idx_train = perm[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
# idx_val = perm[int(len(data_1) * (1 - VALIDATION_SPLIT)):]
#
# data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
# data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
# labels_train = np.concatenate((labels[idx_train], labels[idx_train]))
#
# data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
# data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
# labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

# weight_val = np.ones(len(labels_val))
# if re_weight:
#     weight_val *= 0.472001959
#     weight_val[labels_val == 0] = 1.309028344


########################################
# define the model structure
########################################
def get_model():
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input], \
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model


#######################################
# train the model
########################################


def train_model():
    print(STAMP)
    model = get_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    hist = model.fit([data_1, data_2], labels, \
                     validation_data=([data_1, data_2], labels), \
                     epochs=100, batch_size=10, shuffle=True, callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_score = min(hist.history['loss'])
    bst_acc = max(hist.history['acc'])
    print bst_acc, bst_score


if __name__ == '__main__':
    train_model()

# predicts = model.predict([test_data_1, test_data_2], batch_size=10, verbose=1)

# for i in range(len(test_ids)):
#    print "t1: %s, t2: %s, score: %s" % (test_texts_1[i], test_texts_2[i], predicts[i])
