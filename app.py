# Import libraries
import numpy as np
import pickle, re
import pandas as pd
import sqlite3
import joblib

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.feature_extraction.text import CountVectorizer

# Pakai PySastrawi buat bantu stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Pakai nltk buat ambil stopwords nya
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from, LazyJSONEncoder, LazyString
app = Flask(__name__)

# Swagger template
app.json_encoder = LazyJSONEncoder

swagger_template = {
    "info": {
        "title": 'API documentation for ML and DL',
        "version": "1.0.1",
        "description": "API for sentiment prediction using keras NN, LSTM, and MLPClassifier models",
    },
}
# Swagger config
swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': "/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

# Bikin DB Connection
conn = sqlite3.connect('database/platinum_challenge.db', check_same_thread=False)
cursor = conn.cursor()

# Bikin Tabel di DB
# EP Text
cursor.execute('''CREATE TABLE IF NOT EXISTS LSTM (cleaned_text varchar(255), sentiment varchar(255));''')
cursor.execute('''CREATE TABLE IF NOT EXISTS MLPClassifier (cleaned_text varchar(255), sentiment varchar(255));''')
cursor.execute('''CREATE TABLE IF NOT EXISTS NN (cleaned_text varchar(255), sentiment varchar(255));''')
# EP File
cursor.execute('''CREATE TABLE IF NOT EXISTS LSTM_file (cleaned_text varchar(255), sentiment varchar(255));''')
cursor.execute('''CREATE TABLE IF NOT EXISTS MLPClassifier_file (cleaned_text varchar(255), sentiment varchar(255));''')
cursor.execute('''CREATE TABLE IF NOT EXISTS NN_file (cleaned_text varchar(255), sentiment varchar(255));''')

# Define max_features untuk model
max_features = 100000

# Define tokenizer
tokenizer = Tokenizer(num_words= max_features, split =' ', lower=True)

# Sentiment list
sentiment = ['negative', 'neutral', 'positive']

# Stopword
list_stopword = stopwords.words('indonesian')
list_stopword.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'nya', 'ber', 'banget', 'kali'])
list_stopword = list(dict.fromkeys(list_stopword))
list_stopword = set(list_stopword)

# Ini dibuat soalnya sebelumnya data-data ini masuk ke dalam stopword nltk indonesian
bukan_stopword = {'baik', 'masalah', 'yakin', 'tidak', 'pantas', 'lebih'}
final_stopword = set([word for word in list_stopword if word not in bukan_stopword])

# Stemming pake PySastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Bikin fungsi stemming nya
def stemming(text):
    text = stemmer.stem(text)
    return text

# Bikin fungsi hilangin stopword
def remove_stopword(text):
    return [word for word in text if not word in final_stopword]

# Bikin fungsi cleansing
def cleansing(text):
    url_pattern = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
    text = re.sub(url_pattern, " ", text)
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    # text = re.sub(r'(\d+)', r' \1 ', text)
    text = re.sub(r'(\d+)',"", text)
    text = re.sub(r' {2,}', " ", text)
    text = re.sub(r'\n\t',' ',text)
    text = re.sub(r'user',' ',text)
    text = re.sub('  +', ' ', text)
    text = text.lower()
    return text

# file = open('tokenizer.pickle', 'rb')
# feature_file = pickle.load(file)
# file.close()

# Load model keras
model_nn = load_model('nn_model.keras')
model_lstm = load_model('lstm_model.keras')

# Load model MLPClassifier
mlp_file = open('mlp.pkl', 'rb')
model_mlp = pickle.load(mlp_file)
mlp_file.close()

# Load vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Homepage
@app.route('/')
def home():
    return "Welcome to the model API!"

# End point text prediction pakai model NN
@swag_from('docs/nn.yaml', methods=['POST'])
@app.route('/nn', methods=['POST'])

def nn():
    original_text = request.form.get('text')
    text_stem = [stemming(original_text)]
    text_wsw = [remove_stopword(text_stem)]
    text = [cleansing(str(text_wsw))]

    feature = tokenizer.texts_to_sequences([text])
    X = pad_sequences(feature, maxlen=55)
    prediction = model_nn.predict(X)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    text =''.join(text)

    cursor.execute("INSERT INTO NN (cleaned_text, sentiment) VALUES ('"+ text +"', '"+ get_sentiment +"')")
    conn.commit()

    json_response = {
        'status_code': 200,
        'description': "NN Prediction Result",
        'data': {
            'text': original_text,
            'cleaned_text': text,
            'sentiment': get_sentiment,}
    }
    response_data = jsonify(json_response)
    return response_data

# End point file prediction pakai model NN
@swag_from("docs/nn_file.yaml", methods=['POST'])
@app.route('/nn_file', methods=['POST'])
def nn_file():
    file = request.files.getlist('file')[0]
    colnames = ['text', 'sentiment']
    df = pd.read_csv(file, sep='\t', header=None, names=colnames)
    texts = df['text'].tolist()

    cleaned_text = []
    for text_input in texts:
        text = stemming(text_input)
        text = remove_stopword(text_input)
        text = cleansing(text_input)

        feature = tokenizer.texts_to_sequences([text])
        X = pad_sequences(feature, maxlen=55)
        prediction = model_nn.predict(X)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        cursor.execute("INSERT INTO NN_file (cleaned_text, sentiment) VALUES ('"+ text +"', '"+ get_sentiment +"')")
        conn.commit()
        cleaned_text.append({
            'text': text_input,
            'cleaned_text': text,
            'sentiment': get_sentiment
        })
    json_response = {
    'status_code': 200,
    'description': "NN_file Prediction Result",
    'data': cleaned_text,
    }
    response_data = jsonify(json_response)
    return response_data
        
# End point text prediction pakai model LSTM
@swag_from('docs/lstm.yaml', methods=['POST'])
@app.route('/lstm', methods=['POST'])

def lstm():
    original_text = request.form.get('text')
    text_stem = [stemming(original_text)]
    text_wsw = [remove_stopword(text_stem)]
    text = [cleansing(str(text_wsw))]

    feature = tokenizer.texts_to_sequences(text)
    X = pad_sequences(feature, maxlen=64)
    prediction = model_lstm.predict(X)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    text =''.join(text)

    cursor.execute("INSERT INTO LSTM (cleaned_text, sentiment) VALUES ('"+ text +"', '"+ get_sentiment +"')")
    conn.commit()

    json_response = {
        'status_code': 200,
        'description': "LSTM Prediction Result",
        'data': {
            'text': original_text,
            'cleaned_text': text,
            'sentiment': get_sentiment,}
    }
    response_data = jsonify(json_response)
    return response_data

# End point file prediction pakai model LSTM
@swag_from("docs/lstm_file.yaml", methods=['POST'])
@app.route('/lstm_file', methods=['POST'])
def lstm_file():
    file = request.files.getlist('file')[0]
    colnames = ['text', 'sentiment']
    df = pd.read_csv(file, sep='\t', header=None, names=colnames)
    texts = df['text'].tolist()

    cleaned_text = []
    for text_input in texts:
        text = stemming(text_input)
        text = remove_stopword(text_input)
        text = cleansing(text_input)

        feature = tokenizer.texts_to_sequences([text])
        X = pad_sequences(feature, maxlen=64)
        prediction = model_lstm.predict(X)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        cursor.execute("INSERT INTO LSTM_file (cleaned_text, sentiment) VALUES ('"+ text +"', '"+ get_sentiment +"')")
        conn.commit()
        cleaned_text.append({
            'text': text_input,
            'cleaned_text': text,
            'sentiment': get_sentiment
        })
    json_response = {
    'status_code': 200,
    'description': "LSTM_file Prediction Result",
    'data': cleaned_text,
    }
    response_data = jsonify(json_response)
    return response_data

# End point text prediction pakai model MLP Classifier
@swag_from('docs/mlp.yaml', methods=['POST'])
@app.route('/mlp', methods=['POST'])

def mlp():
    original_text = request.form.get('text')
    text_stem = [stemming(original_text)]
    text_wsw = [remove_stopword(text_stem)]
    text = [cleansing(str(text_wsw))]

    X = vectorizer.transform(text)
    prediction = model_mlp.predict(X)
    predicted = list(prediction)
    id2label = {0: 'neutral', 1: 'positive', 2: 'negative'}
    get_sentiment = list(map(id2label.get, predicted))

    text =''.join(text)
    get_sentiment =''.join(get_sentiment)

    cursor.execute("INSERT INTO MLPClassifier (cleaned_text, sentiment) VALUES ('"+ text +"', '"+ get_sentiment +"')")
    conn.commit()

    json_response = {
        'status_code': 200,
        'description': "MLP Classifier Prediction Result",
        'data': {
            'text': original_text,
            'cleaned_text': text,
            'sentiment': get_sentiment,}
    }
    response_data = jsonify(json_response)
    return response_data

# End point file prediction pakai model MLP Classifier
@swag_from("docs/mlp_file.yaml", methods=['POST'])
@app.route('/mlp_file', methods=['POST'])
def mlp_file():
    file = request.files.getlist('file')[0]
    colnames = ['text', 'sentiment']
    df = pd.read_csv(file, sep='\t', header=None, names=colnames)
    texts = df['text'].tolist()

    cleaned_text = []
    for text_input in texts:
        text = [stemming(text_input)]
        text = [remove_stopword(text)]
        text = [cleansing(str(text))]

        X = vectorizer.transform(text)
        prediction = model_mlp.predict(X)
        predicted = list(prediction)
        id2label = {0: 'neutral', 1: 'positive', 2: 'negative'}
        get_sentiment = list(map(id2label.get, predicted))

        text =''.join(text)
        get_sentiment =''.join(get_sentiment)

        cursor.execute("INSERT INTO LSTM_file (cleaned_text, sentiment) VALUES ('"+ text +"', '"+ get_sentiment +"')")
        conn.commit()
        cleaned_text.append({
            'text': text_input,
            'cleaned_text': text,
            'sentiment': get_sentiment
        })
    json_response = {
    'status_code': 200,
    'description': "LSTM_file Prediction Result",
    'data': cleaned_text,
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()