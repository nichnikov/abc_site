#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:44:01 2018

@author: alexey
mystem python wrap
https://pypi.org/project/pymystem3/
"""

import Classifier_tag22_func, keras, os
from flask import Flask, request
import tensorflow as tf
#from pymystem3 import Mystem

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])#, 'POST']) #allow both GET and POST requests
def form_example():
    if request.method == 'POST': #this block is only entered when the form is submitted
        #text = request.form.get('form_name')
        text = request.form.get('form_name')
        result = Classifier_tag22_func.classifier_cnn(text, cnn_classifier, graph_defaut)
        
        #m = Mystem()
        #lemmas = m.lemmatize(text)
        #ret_txt = ' '.join(lemmas)

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
    model_rout = r'/home/an/Dropbox/abc_site/models'
    dict_rout = r'/home/an/Dropbox/abc_site/dicts'


    text  = 'Раскрутка сайта www.c-vertical.ru.Занимаюсь подбором ключевых словосочетаний для раскрутки сайта. Подскажите из скольки слов должно быть словосочетание. Например: &quot;строительство зданий из'
    global graph_defaut
    graph_defaut = tf.get_default_graph()
    global cnn_classifier
    cnn_classifier = keras.models.load_model(os.path.join(model_rout, 'cnn_model_abc_site_20181130_tensor150.h5'))
    
    app.run(debug=True, host='127.0.0.1', port=5003)
     
