#%%
import os, pickle, re, glob
import pandas as pd
from functions import text_coll_lemmatizer, data_for_learning_selection, texts_change_patterns, tag_handling
from functions import make_embedded_sequences
import keras
from keras.models import load_model
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers.core import SpatialDropout1D
from keras import regularizers
import numpy as np
from random import shuffle
from gensim import models
from gensim.models import Doc2Vec


model_rout = r'./models'
data_rout = r'./data/'
dict_rout = r'./dicts'

#выгрузим размеченные тексты
ds = pd.DataFrame(pd.read_csv(os.path.join(data_rout, 'txts_lbls.csv')))

#отберем те классы примеров, для которых есть модели:
tag_nums = []
for fn in glob.glob(os.path.join(model_rout, '*.h5')):
    tag_num = re.findall('\d*_', fn)
    if tag_num !=[]:
        tag_nums.append(re.sub('_', '', tag_num[0]))


txts_for_initial_learning_df = ds[~ds['lbs'].isin(tag_nums)]
txts_for_adding_learning_df = ds[ds['lbs'].isin(tag_nums)]
print ('Hello, Ia m ready!')

#%%
#print(txts_for_initial_learning_df)
print(txts_for_adding_learning_df)

#%%
def data_for_learning_selection(ds, each_class_examples=100):
    class_q = pd.DataFrame(ds['txt'].groupby(ds['lbs']).count())
    class_q.reset_index(inplace = True)
    class_for_learning = class_q[class_q['txt'] >=each_class_examples]
    ds_lbs_for_learn = ds.loc[ds['lbs'].isin(list(class_for_learning['lbs']))]
    return  list(ds_lbs_for_learn['txt']), list(ds_lbs_for_learn['lbs'])


#отберем те классы примеров, в которых больше n (например, n=100) текстов:
txts, lbls = data_for_learning_selection(txts_for_adding_learning_df, each_class_examples=100)
#ds_lbs_for_learn = data_for_learning_selection(txts_for_adding_learning_df, each_class_examples=100)

#%%
print(txts[:10])

#%%
#проведем некоторые замены (перечисляются в списке кортежей):
patterns = [(r'www\.\w+\.\w+|\w+\.\w+\.рф', 'ссылканасайт '), ('[aA-zZ]', ' '),
            (r'[^\w\s]', ''), (r'[\d+]', ''), (r'\s+', ' ')]

txts_ = texts_change_patterns(txts, patterns)
print(txts_[:10])

#%%
#лемматизация текстов:
lemm_txts = text_coll_lemmatizer(txts_)

print(lemm_txts[:10])

#%%
#загрузка w2v модели
#w2v_model = Doc2Vec.load(os.path.join(model_rout, 'w2v_model'))
w2v_model = models.Word2Vec([tx.split() for tx in lemm_txts], 
                             min_count=30, iter=25, size=150, window=5,  workers=7)

#дообучение w2v модели

#обновление словаря:
work_vocab = [w for w in w2v_model.wv.vocab]

#сохранение дообученной w2v модели
texts_for_working = [[w for w in sen.split() if w in work_vocab] for sen in lemm_txts]

#%%
#загрузим токенайзер
with open(os.path.join(model_rout, 'tokenizer500words_lstm_embedding.pickle'), 'rb') as f:
    tokenizer = pickle.load(f)

#добавим в токенайзер новые слова (дообучим токенайзер)


#объединим тексты с тегами:
texts_with_lables = list(zip(texts_for_working, lbls))

#%%
print(txts[:100])
#txts_for_adding_learning_df

#print(lemm_txts)
#print(lbls)
#print(texts_for_working)
#print(texts_with_lables)

#%%
#для одного тега построим дообучение модели:
print(set(lbls))
for tag_num in set(lbls):
    print(tag_num)

    txs0, txs1 = tag_handling(texts_with_lables, tag_num, balance = True)
    print(txs0[:10])

    texts_for_learning = txs0 + txs1
    lbs = len(txs0)*[0] + len(txs1)*[1]  
    print('Создаем embedding_matrix')

    #подготовка данных для обучения моделей:
    labled_texts = list(zip(texts_for_learning, lbs))
    shuffle(labled_texts)
    txts_for_learning, lbs_for_learning = zip(*labled_texts)

    MAX_SEQUENCE_LENGTH = 15
    embedded_sequences, comment_input = make_embedded_sequences(tokenizer, w2v_model, work_vocab, MAX_SEQUENCE_LENGTH)

    num_classes = len(set(lbs_for_learning))    

    # pad
    sequences = [[tokenizer.word_index.get(t, 0) for t in comment] for comment in txts_for_learning]    

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
    y = keras.utils.to_categorical(np.array(lbs_for_learning), num_classes)
    print(data.shape)

    #загрузим готовую модель
    model_name = str(tag_num) + '_lstm_model.h5'
    nn_model = load_model(os.path.join(model_rout, model_name))

    #дообучим модель:
    nn_model.fit([data], y, validation_split=0.2, epochs=5, batch_size=32, shuffle=True)
    nn_model.save(os.path.join(model_rout, model_name))
