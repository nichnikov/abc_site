#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:44:01 2018

@author: alexey
mystem python wrap
https://pypi.org/project/pymystem3/
"""
import Classifier_multitags_func, keras, os, glob, re
from flask import Flask, request
import tensorflow as tf

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])#, 'POST']) #allow both GET and POST requests
def form_example():
    if request.method == 'POST': #this block is only entered when the form is submitted
        text = request.form.get('form_name')
        #ts = Classifier_multitags_func.text2tensor(text)
        result = Classifier_multitags_func.classifier_cnn(text, cnn_classifier_list, tags_list, graph_defaut)
        
        return '''</h5>для запроса: {} </h5> <br>
                    </h3> Class: {} </h3>'''.format(text, result)

    return '''
            <form method="POST">
               <p>Текст для классификации<Br>
                 <textarea name="form_name" cols="40" rows="3"></textarea></p>
                 <p><input type="submit" value="Отправить">
            </form>
          '''
#форма взята вот отсюда: http://htmlbook.ru/samhtml5/formy/otpravka-dannykh-formy

if __name__ == '__main__':
    model_rout = r'./models'
    dict_rout = r'./dicts'


    text  = 'Раскрутка сайта www.c-vertical.ru.Занимаюсь подбором ключевых словосочетаний для раскрутки сайта. Подскажите из скольки слов должно быть словосочетание. Например: &quot;строительство зданий из'
    global graph_defaut
    graph_defaut = tf.get_default_graph()
    global cnn_classifier_list    
    cnn_classifier_list = []
    global tags_list
    tags_list = []
    
    for f_cnn in glob.glob(os.path.join(model_rout, '*.h5')):
        tags_list.append(re.findall('tag\d+', f_cnn)[0])
        cnn_classifier_list.append(keras.models.load_model(f_cnn))

    #cnn_classifier = keras.models.load_model(os.path.join(model_rout, 'cnn_model_abc_site_20181130_tensor150.h5'))   
    app.run(debug=True, host='0.0.0.0', port=4999)
     
