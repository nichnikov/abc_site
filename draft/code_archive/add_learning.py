import os, pickle, re, glob
import pandas as pd
from functions import text_coll_lemmatizer, data_for_learning_selection, texts_change_patterns, tag_handling
from functions import make_embedded_sequences, adding_nn_lerning
from keras.preprocessing.text import Tokenizer
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

#отберем те классы примеров, в которых больше n (например, n=100) текстов:
txts, lbls = data_for_learning_selection(txts_for_adding_learning_df, each_class_examples=100)

#проведем некоторые замены (перечисляются в списке кортежей):
patterns = [(r'www\.\w+\.\w+|\w+\.\w+\.рф', 'ссылканасайт '), ('[aA-zZ]', ' '),
            (r'[^\w\s]', ''), (r'[\d+]', ''), (r'\s+', ' ')]

txts_ = texts_change_patterns(txts, patterns)

#лемматизация текстов:
lemm_txts = text_coll_lemmatizer(txts_)

#загрузка w2v модели:
w2v_model = Doc2Vec.load(os.path.join(model_rout, 'w2v_model'))
#дообучение w2v модели:
w2v_model.train([tx.split() for tx in lemm_txts], total_examples = w2v_model.corpus_count, epochs=10)

#сохранение дообученной w2v модели:
w2v_model.save(os.path.join(model_rout, 'w2v_model'))

#обновление словаря:
work_vocab = [w for w in w2v_model.wv.vocab]

#отберем в текстах те слова, которые есть в word2vec модели:
texts_for_working = [[w for w in sen.split() if w in work_vocab] for sen in lemm_txts]

#загрузим токенайзер
with open(os.path.join(model_rout, 'tokenizer500words_lstm_embedding.pickle'), 'rb') as f:
    tokenizer = pickle.load(f)

#добавим в токенайзер новые слова (дообучим токенайзер)
tokenizer.fit_on_texts(texts_for_working)
#сохраним обновленный токенайзер:
with open(os.path.join(model_rout, 'tokenizer.pickle'), 'wb') as file:
        pickle.dump(tokenizer, file)
    

#объединим тексты с тегами:
texts_with_lables = list(zip(texts_for_working, lbls))

print('все теги:', set(lbls))
lables_nums = set(lbls)

#для одного тега построим дообучение модели:
lables_nums = {6}
MAX_SEQUENCE_LENGTH = 15

#функция дообучения модели по указанным тегированным текстам:
adding_nn_lerning(model_rout, lables_nums, texts_with_lables, tokenizer, w2v_model, work_vocab, MAX_SEQUENCE_LENGTH, balance_type = True, epochs_num=3)