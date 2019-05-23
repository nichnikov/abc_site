import os, io, pickle, re, glob, argparse
import pandas as pd
from functions import new_nn_learning, nn_updating_new_data
from flask import Flask, request, Response, send_file, make_response

app = Flask(__name__)

#страница загрузки
@app.route('/')
def index():
    return send_file(os.path.join(data_rout, 'file_loading_form.html'))

# Страница, куда постим CSV файл для обработки
@app.route('/', methods=[ 'POST' ])
def nn_lerning():
    #выгрузим размеченные тексты
    file = request.files['f']
    ds = pd.DataFrame(pd.read_csv(file))

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
    new_nn_learning(model_rout, data_rout, dict_rout, txts_for_initial_learning_df, nn_epochs=3)

    #дообучим модели по классам, по которым модели уже существуют:
    df_test = txts_for_adding_learning_df[txts_for_adding_learning_df['lbs'].isin([6, 13])]
    nn_updating_new_data(df_test, model_rout, nn_epochs=3)
    
    mf = io.BytesIO()
    #mf = io.StringIO()
    mf.write('I have done job!'.encode('utf-8'))
    mf.seek(0)
    
    return send_file(mf, attachment_filename='report.txt', as_attachment=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Классификатор')
    parser.add_argument('--host', dest='host',  default='0.0.0.0')
    parser.add_argument('--port', dest='port', default=5001)
    args=parser.parse_args()
    global model_rout
    model_rout = r'./models'
    global data_rout
    data_rout = r'./data/'
    global dict_rout
    dict_rout = r'./dicts'
    
    app.run(debug=True, host=args.host, port=int(args.port))