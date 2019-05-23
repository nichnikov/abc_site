
import os, pickle, re, glob
import pandas as pd
from functions import new_nn_learning, nn_updating_new_data

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


#выберем те тексты, для которых моделей нет и модели по которым надо будет создать и обучить:
txts_for_initial_learning_df = ds[~ds['lbs'].isin(tag_nums)]
#выберем те тексты, по которым надо будет модели дообучить:
txts_for_adding_learning_df = ds[ds['lbs'].isin(tag_nums)]

print(txts_for_adding_learning_df[txts_for_adding_learning_df['lbs'].isin([6, 13])])
set(list(txts_for_adding_learning_df['lbs']))

#создадим модели для классов, по которым еще нет моделей:
new_nn_learning(model_rout, data_rout, dict_rout, txts_for_initial_learning_df, nn_epochs=5)

#дообучим модели по классам, по которым модели уже существуют:
df_test = txts_for_adding_learning_df[txts_for_adding_learning_df['lbs'].isin([6, 13])]
nn_updating_new_data(df_test, model_rout, nn_epochs=5)