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

app = Flask(__name__)
cors = CORS(app)



def predictiohelper(inputimg):
    img = image.load_img(inputimg, target_size=(128, 128))
    inputimg = image.img_to_array(img)
    x = np.expand_dims(inputimg, axis=0)
    model = pickle.load(open('finalized_model.sav', 'rb'))
    predction = model.predict(x)

    return predction


@app.route("/")
def index():
    return "Api is working go to /api/predict to get predction with img as input"


@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predclass = {0: 'Parasitized', 1: 'Uninfected'}

        predictions = predictiohelper(file_path)
    # print('INFO Predictions: {}'.format(predictions))
        return jsonify(predclass[np.argmax(predictions)])
    # return jsonify(data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port)
