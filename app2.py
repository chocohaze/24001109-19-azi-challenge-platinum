from types import new_class
import numpy as np, h5py
import pickle, re

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import h5py as h5

from sklearn.feature_extraction.text import CountVectorizer

# Pakai PySastrawi buat bantu stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Pakai nltk buat ambil stopwords nya
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import os.path

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
    # 'host' : LazyString(lambda: request.host)
    # "host": "localhost:5000",
    # "schemes": ["http"],
}
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

max_features = 100000
tokenizer = Tokenizer(num_words= max_features, split =' ', lower=True)
sentiment = ['negative', 'neutral', 'positive']

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

def words_to_sentence(list_words):
    return ' '.join(list_words)

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

file = open('tokenizer.pickle', 'rb')
feature_file = pickle.load(file)
file.close()

model_nn = load_model('nn_model.keras')
model_lstm = load_model('lstm_model.keras')

mlp_file = open('mlp.pkl', 'rb')
model_mlp = pickle.load(mlp_file)

# # Function to load pickle files with version handling
# def load_pickle(file_path):
#     try:
#         with open(file_path, 'rb') as handle:
#             return pickle.load(handle)
#     except Exception as e:
#         logging.error(f"Failed to load pickle file {file_path}: {str(e)}")
#         return None

# # Load necessary files
# tokenizer = load_pickle('tokenizer.pickle')
# text_preprocessing = load_pickle('text_preprocessing.pickle')
# onehot_encode = load_pickle('onehot_encode.pickle')

# # Ensure all necessary objects are loaded
# if tokenizer is None or text_preprocessing is None or onehot_encode is None:
#     raise RuntimeError("Failed to load one or more necessary files. Check logs for details.")

# # Preprocessing function
# def cleansing(text):
#     return text_preprocessing(text)

# def load_model_nn(model_path):
#     try:
#         model = tf.keras.models.load_model(model_path)
#         return model
#     except Exception as e:
#         logging.error(f"Failed to load NN model: {str(e)}")
#         return str(e)

# def load_model_lstm(model_path):
#     try:
#         model = tf.keras.models.load_model(model_path)
#         return model
#     except Exception as e:
#         logging.error(f"Failed to load LSTM model: {str(e)}")
#         return str(e)

# def classify_text(model, text):
#     cleaned_text = cleansing(text)
#     sequences = tokenizer.texts_to_sequences([cleaned_text])
#     padded_sequences = pad_sequences(sequences, maxlen=55)
#     prediction = model.predict(padded_sequences)
#     sentiment = np.argmax(prediction, axis=1)[0]
    
#     if sentiment == 1:
#         return "positive"
#     elif sentiment == 0:
#         return "negative"
#     else:
#         return "neutral"

@app.route('/')
def home():
    return "Welcome to the model API!"

@swag_from('docs/nn.yaml', methods=['POST'])
@app.route('/nn', methods=['POST'])

def nn():
    original_text = request.form.get('text')
    text_stem = [stemming(original_text)]
    text_wsw = [remove_stopword(text_stem)]
    text = [cleansing(str(text_wsw))]
    feature = tokenizer.texts_to_sequences(text)
    X = pad_sequences(feature, maxlen=55)
    prediction = model_nn.predict(X)
    get_sentiment = sentiment[np.argmax(prediction[0])]

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

@swag_from('docs/mlp.yaml', methods=['POST'])
@app.route('/mlp', methods=['POST'])

def mlp():
    original_text = request.form.get('text')
    text_stem = [stemming(original_text)]
    text_wsw = [remove_stopword(text_stem)]
    text = [cleansing(str(text_wsw))]
    vectorizer = CountVectorizer(decode_error='ignore', lowercase=True, min_df=1, max_df=2)
    vectorized = vectorizer.fit(['text'])#new_class['text'].values.astype('U'))
    trans = vectorizer.fit_transform(vectorized)
    X = trans.toarray()
    # X = pad_sequences(feature, maxlen=64)
    prediction = model_mlp.predict(X)
    get_sentiment = sentiment[np.argmax(prediction[0])]

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
if __name__ == '__main__':
    app.run()

#         file = request.files['file']
#         file.save('nn_model.h5')
#         logging.debug(f"Received file for NN model: {file.filename}")
        
#         # Process text and file as needed
#         model = load_model_nn('nn_model.h5')
        
#         if isinstance(model, str):
#             logging.error(f"Error loading NN model: {model}")
#             return jsonify({"error": model}), 400
        
#         sentiment = classify_text(model, text)
        
#         return jsonify({"message": "NN model uploaded successfully", "text": text, "sentiment": sentiment})
#     except Exception as e:
#         logging.error(f"Error in /upload_nn: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/upload_lstm', methods=['POST'])
# @swag_from('upload_lstm.yaml')
# def upload_lstm():
#     try:
#         text = request.form.get('text')
#         file = request.files['file']
#         file.save('lstm_model.h5')
#         logging.debug(f"Received file for LSTM model: {file.filename}")
        
#         # Process text and file as needed
#         model = load_model_lstm('lstm_model.h5')
        
#         if isinstance(model, str):
#             logging.error(f"Error loading LSTM model: {model}")
#             return jsonify({"error": model}), 400
        
#         sentiment = classify_text(model, text)
        
#         return jsonify({"message": "LSTM model uploaded successfully", "text": text, "sentiment": sentiment})
#     except Exception as e:
#         logging.error(f"Error in /upload_lstm: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
