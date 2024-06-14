from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from flasgger import Swagger, swag_from
import logging
import numpy as np
import pickle
import re
import os
import tensorflow as tf

app = Flask(__name__)

swagger_template = {
    "info": {
        "title": "Model API",
        "description": "API for uploading NN and LSTM models",
        "version": "0.0.1"
    },
    "host": "localhost:5000",
    "schemes": ["http"],
}
# swagger_config = {
#     'headers': [],
#     'specs': [
#         {
#             'endpoint': 'apidocs',
#             'route': '/apidocs.json',
#         }
#     ],
#     'static_url_path': '/flasgger_static',
#     'swagger_ui': True,
#     'specs_route': "/apidocs/"
# }
swagger = Swagger(app, template=swagger_template)#, config=swagger_config)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# Function to load pickle files with version handling
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        logging.error(f"Failed to load pickle file {file_path}: {str(e)}")
        return None

# Load necessary files
tokenizer = load_pickle('tokenizer.pickle')
text_preprocessing = load_pickle('text_preprocessing.pickle')
onehot_encode = load_pickle('onehot_encode.pickle')

# Ensure all necessary objects are loaded
if tokenizer is None or text_preprocessing is None or onehot_encode is None:
    raise RuntimeError("Failed to load one or more necessary files. Check logs for details.")

# Preprocessing function
def cleansing(text):
    return text_preprocessing(text)

def load_model_nn(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        logging.error(f"Failed to load NN model: {str(e)}")
        return str(e)

def load_model_lstm(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        logging.error(f"Failed to load LSTM model: {str(e)}")
        return str(e)

def classify_text(model, text):
    cleaned_text = cleansing(text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=55)
    prediction = model.predict(padded_sequences)
    sentiment = np.argmax(prediction, axis=1)[0]
    
    if sentiment == 1:
        return "positive"
    elif sentiment == 0:
        return "negative"
    else:
        return "neutral"

@app.route('/')
def home():
    return "Welcome to the model API!"

@app.route('/upload_nn', methods=['POST'])
@swag_from('upload_nn.yaml')
def upload_nn():
    try:
        text = request.form.get('text')
        file = request.files['file']
        # file.save('nn_model.keras')
        logging.debug(f"Received file for NN model: {file.filename}")
        
        # Process text and file as needed
        model = load_model_nn('nn_model.keras')
        
        if isinstance(model, str):
            logging.error(f"Error loading NN model: {model}")
            return jsonify({"error": model}), 400
        
        sentiment = classify_text(model, text)
        
        return jsonify({"message": "NN model uploaded successfully", "text": text, "sentiment": sentiment})
    except Exception as e:
        logging.error(f"Error in /upload_nn: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_lstm', methods=['POST'])
@swag_from('upload_lstm.yaml')
def upload_lstm():
    try:
        text = request.form.get('text')
        file = request.files['file']
        # file.save('lstm_model.keras')
        logging.debug(f"Received file for LSTM model: {file.filename}")
        
        # Process text and file as needed
        model = load_model_lstm('lstm_model.keras')
        
        if isinstance(model, str):
            logging.error(f"Error loading LSTM model: {model}")
            return jsonify({"error": model}), 400
        
        sentiment = classify_text(model, text)
        
        return jsonify({"message": "LSTM model uploaded successfully", "text": text, "sentiment": sentiment})
    except Exception as e:
        logging.error(f"Error in /upload_lstm: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
