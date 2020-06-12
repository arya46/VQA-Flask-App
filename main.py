import numpy as np
import pandas as pd
import re
import glob
from flask import Flask, request, render_template, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
logging.basicConfig(level=logging.INFO)

import tensorflow as tf
import silence_tensorflow.auto  # pylint: disable=unused-import
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import sequence

from models.arch import build_model
from models.layers import ContextVector, PhraseLevelFeatures, AttentionMaps
from utils.load_pickles import tok, labelencoder
from utils.helper_functions import image_feature_extractor, process_sentence, predict_answers

max_answers = 1000
max_seq_len = 22
vocab_size  = len(tok.word_index) + 1
dim_d       = 512
dim_k       = 256
l_rate      = 1e-4
d_rate      = 0.5
reg_value   = 0.01
MODEL_PATH = 'pickles/complete_model.h5'
IMAGE_PATH = 'static'

custom_objects = {
    'PhraseLevelFeatures': PhraseLevelFeatures,
    'AttentionMaps': AttentionMaps,
    'ContextVector': ContextVector
    }

# load the model
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
vgg_model = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(3, 224, 224)))
  
# Create Flask application
app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        
        # delete images uploaded in previous session
        files = glob.glob(IMAGE_PATH+'/*')
        for f in files:
           os.remove(f)
           
        #0 --- Get the image file and question
        f = request.files['image_file']
        fname = secure_filename(f.filename)
        f.save(IMAGE_PATH +'/'+ fname)
        question = request.form["question"]
        
        #1 --- Extract image features
        img_feat = image_feature_extractor(IMAGE_PATH +'/'+ fname, vgg_model)
    
        #2 --- Clean the question
        questions_processed = pd.Series(question).apply(process_sentence)
        
        
        #3 --- Tokenize the question data using a pre-trained tokenizer and pad them
        question_data = tok.texts_to_sequences(questions_processed)
        question_data = sequence.pad_sequences(question_data, \
                                               maxlen=max_seq_len,\
                                               padding='post')
    
    
        #4 --- Predict the answers
        y_predict = predict_answers(img_feat, question_data, model, labelencoder)
        return render_template('index.html', fname=fname, question=question, answer=y_predict[0])
    
    except Exception as e:
        
        return render_template("error.html" , error = e)
    
# RUN FLASK APPLICATION
if __name__ == "__main__":

    # RUNNNING FLASK APP    
    app.run(debug=False, host = '0.0.0.0', port=8080)



