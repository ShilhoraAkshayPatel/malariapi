from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import pickle
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import os
from werkzeug.utils import secure_filename
import base64
from scipy.misc import imsave, imread, imresize
import re

app = Flask(__name__)
cors = CORS(app)


def predictiohelper(inputimg):
    model = pickle.load(open('finalized_model.sav', 'rb'))
    predction = model.predict(inputimg)

    return predction


def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route("/")
def index():
    return "Api is working go to /api/predict to get predction with img as input"


@app.route('/api/predict', methods=['POST'])
def predict():
    f = request.files['file']
    convertImage(f)
    xx = imread('output.png', mode='L')
    xxx = np.invert(xx)
    xxxx = imresize(xxx, (128, 128))
    xxxxx = xxxx.reshape(1, 128, 128, 3)

    predclass = {0: 'Parasitized', 1: 'Uninfected'}

    predictions = predictiohelper(xxxxx)
    # print('INFO Predictions: {}'.format(predictions))
    return jsonify(predclass[np.argmax(predictions)])
    # return jsonify(data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port)
