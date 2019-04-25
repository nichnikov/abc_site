#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:47:41 2019

@author: alexey
"""

import os, pickle
from gensim import models, logging


model_rout = r'./models'
data_rout = r'./data/'
dict_rout = r'./dicts'

with open (os.path.join(data_rout, 'lebled_texts.pickle'), 'rb') as f:
    data = pickle.load(f)

txts = [i[0] for i in data]

#создание word2vec модели
w2v_model = models.Word2Vec([tx.split() for tx in txts], min_count=30, iter=25, size=150, window=5,  workers=7)

work_vocab = [w for w in w2v_model.wv.vocab]
w2v_model.save(os.path.join(model_rout, 'w2v_model_abc_tech_20190214'))


txts, lbls = zip(*data)

lemm_texts_true_words = [[w for w in sen.split() if w in work_vocab] for sen in txts]
w2v = models.Word2Vec.load(os.path.join(model_rout, 'w2v_model_abc_tech_20190214'))
#w2v.wv.most_similar('ссылканасайт')


#подготовка текстов
'''нужно тексты с определенным тегом (например, с 22) объявить текстом с одним тегом (с 1) и удалить все случаи, где эти тексты встречаются с другими тегами (с 0)'''
#для 22 лейбла
tx_lbs = list(zip(lemm_texts_true_words, lbls))
txt_22 = [i[0] for i in tx_lbs if i[1] == 22]
txt_no22 = [i[0] for i in tx_lbs if i[1] != 22]
txt_no22 = [i for i in txt_no22 if i not in txt_22]

for i in txt_22:
    print(i)


#создание моделей на основании нейронных сетей:
#формирование векторов (тензоров):
import keras
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers.core import SpatialDropout1D
from keras import regularizers






top_words = 5000
tokenizer = Tokenizer(num_words = top_words, oov_token=None)
tokenizer.fit_on_texts(texts_for_learning)

#сохранение токенезатора модели
with open(os.path.join(model_rout, 'tokenizer5000words_lstm_embedding.pickle'), 'wb') as file:
    pickle.dump(tokenizer, file)

k=0
for w in tokenizer.word_index.items():
    print(w)
    if k > 5:
        break
    k+=1

print('Создаем embedding_matrix')
embeddings_index = {}

#tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
for word in tokenizer.word_index:
    embeddings_index[word] = model_gensim.wv[word]

embeddings_index['просьба']
embeddings_index.get('просьба')

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
embedding_matrix.shape

MAX_SEQUENCE_LENGTH = 15
for word, i in tokenizer.word_index.items():    
    embedding_vector = embeddings_index.get(word)    
    embedding_matrix[i] = embedding_vector

nb_words = len(tokenizer.word_index) + 1
WV_DIM = EMBEDDING_DIM
wv_matrix = embedding_matrix
wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)

# Inputs
comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = wv_layer(comment_input)

# biGRU
embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
#x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(embedded_sequences)
#x = LSTM(100)(embedded_sequences)
x = LSTM(1000, dropout = 0.1, activation='tanh', recurrent_dropout=0.1, 
         kernel_regularizer=regularizers.l2(0.02), return_sequences=False)(embedded_sequences)

# Output
x = Dense(500)(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)

num_classes = len(set(lbs_for_learning))
preds = Dense(num_classes, activation='sigmoid')(x)#num_classes

# build the model
model = Model(inputs=[comment_input], outputs=preds)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.09, nesterov=True), metrics=['accuracy'])


# pad
#sequences = [[model_gensim.wv[word] for word in sen] for sen in texts_for_learning]
sequences = [[tokenizer.word_index.get(t, 0) for t in comment] for comment in texts_for_learning]

print(len(sequences))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
y = keras.utils.to_categorical(np.array(lbs_for_learning), num_classes)
history = model.fit([data], y, validation_split=0.2, epochs=10, batch_size=32, shuffle=True)

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


