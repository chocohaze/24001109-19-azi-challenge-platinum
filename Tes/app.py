from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flasgger import Swagger, swag_from

app = Flask(__name__)

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Model API",
        "description": "API for uploading NN and LSTM models",
        "version": "0.0.1"
    },
    "host": "localhost:5000",
    "schemes": ["http"],
}

swagger = Swagger(app, template=swagger_template)

# Function to load the NN model
def load_model_nn(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        return str(e)

# Function to load the LSTM model
def load_model_lstm(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return "Welcome to the model API!"

@app.route('/upload_nn', methods=['POST'])
@swag_from('upload_nn.yml')
def upload_nn():
    try:
        text = request.form.get('text')
        file = request.files['file']
        file.save('nn_model.h5')
        
        # Process text and file as needed
        model = load_model_nn('nn_model.h5')
        
        if isinstance(model, str):
            return jsonify({"error": model}), 400
        
        return jsonify({"message": "NN model uploaded successfully", "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_lstm', methods=['POST'])
@swag_from('upload_lstm.yml')
def upload_lstm():
    try:
        text = request.form.get('text')
        file = request.files['file']
        file.save('lstm_model.h5')
        
        # Process text and file as needed
        model = load_model_lstm('lstm_model.h5')
        
        if isinstance(model, str):
            return jsonify({"error": model}), 400
        
        return jsonify({"message": "LSTM model uploaded successfully", "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
