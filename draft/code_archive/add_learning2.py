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

nn_updating_new_data(txts_for_adding_learning_df, model_rout)