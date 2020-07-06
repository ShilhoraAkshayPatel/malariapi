from flask_cors import CORS
import base64
import numpy as np
import io
#from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from flask import request
from flask import jsonify
from flask import Flask
import pickle
import tensorflow as tf

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('finalized_model.sav', 'rb'))


def preprocess_image(image, target_size):
    #if image.mode != "RGB":
        #image = image.convert("RGB")
    #image = image.resize(target_size)
    #image = img_to_array(image)
    #image = np.expand_dims(image, axis=0)
    return image


@app.route("/")
def index():
    return "Api is working go to /api/predict to get predction with img as input"


@app.route('/api/predict', methods=['POST'])
def predict():
    #message = request.get_json(force=True)
    f = request.files['file']
    image=f.filename
    #encoded = message['image']
    #decoded = base64.b64decode(encoded)
    #image = Image.open(io.BytesIO(decoded))
    img = image.load_img(image, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    
    #processed_image = preprocess_image(x)
    #processed_image = preprocess_image(image, target_size=(128, 128))
    prediction = model.predict(x)
    #prediction = model.predict(processed_image)
    preclass = {0: "Parasitized", 1: "Uninfected"}
    response = preclass[np.argmax(prediction)]
    return jsonify(response)


if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 5000))
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
