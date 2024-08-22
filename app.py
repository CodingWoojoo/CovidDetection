from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
import numpy as np
from keras.preprocessing import image
from io import BytesIO
from model import build_resnet50  # Import the function from model.py
from flask.helpers import send_from_directory
import matplotlib.pyplot as plt
from os.path import join, dirname, realpath
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)
disease_class=['Covid-19','Non Covid-19']

# Load your model
model = build_resnet50()
model.load_weights('model.keras')

def data_processing(img_path):
    x = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x
    
def predict_image(img_path):  
    x = data_processing(img_path)     
    prediction = model.predict(x)
    a = prediction[0]
    ind = np.argmax(a)     
    result = disease_class[ind]
    return result

def accuracy_image(img_path):
    x = data_processing(img_path)        
    prediction = model.predict(x)
    a = prediction[0]
    final = max(a)
    return final

@app.route('/')
def index():
    return render_template('index.html', prediction="Covid", accuracy="99.99%", image="..Covid.png")

app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/submit', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = request.files['image_input']
        img_filename = secure_filename(img.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        print(img_path)
        img.save(img_path)
        
        # Generate prediction and accuracy
        final_prediction = predict_image(img_path)
        final_accuracy = accuracy_image(img_path)

        return render_template("index.html", prediction=final_prediction, accuracy=final_accuracy, image=img_filename)


if __name__ == '__main__':
    app.run(debug=True)
