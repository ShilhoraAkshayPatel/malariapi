from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import pickle
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import os
from werkzeug.utils import secure_filename
import base64
from scipy.misc import imsave, imread, imresize
import re
from PIL import Image

app = Flask(__name__)
cors = CORS(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'static')


def predictiohelper(inputimg):
    model = pickle.load(open('finalized_model.sav', 'rb'))
    predction = model.predict(inputimg)

    return predction


@app.route("/")
def index():
    return "Api is working go to /api/predict to get predction with img as input"


@app.route('/api/predict', methods=['POST'])
def predict():
    image = request.files['input_file']
    filename = image.filename
    file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
    image_pil = Image.open(image)
    image_pil.thumbnail((600, 300), Image.ANTIALIAS)
    image_pil.save(file_path)

    # classify image
    image = load_img(image, target_size=(128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    #image = preprocess_input(image)

    predclass = {0: 'Parasitized', 1: 'Uninfected'}

    predictions = predictiohelper(image)
    # print('INFO Predictions: {}'.format(predictions))
    return jsonify(predclass[np.argmax(predictions)])
    # return jsonify(data)


if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 5000))
    app.run()
