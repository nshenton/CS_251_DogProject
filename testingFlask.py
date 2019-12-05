from flask import Flask, jsonify, request
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from sklearn.decomposition import PCA
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

ourModel = load_model('bestModel.h5')
pca = pickle.load( open( "pca.p", "rb" ) )

@app.route('/home', methods=['POST'])
def home():
    data = request.files['fileToUpload']
    b = BytesIO(data.read())
    im = Image.open(b).convert('RGB')
    img = im.resize((224, 224), Image.ANTIALIAS)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    ##process the image and decode the predictions into a vector
    model = ResNet50()
    preds = model.predict(x)
    listPreds = decode_predictions(preds, top=4)[0]
    predictionVector = np.array((listPreds[0][2],listPreds[1][2],listPreds[2][2],listPreds[3][2]))
    returnString = np.array((listPreds[0][1],listPreds[1][1],listPreds[2][1]))
    print(returnString)

    print(predictionVector)
    X = pca.transform(np.array([predictionVector]).reshape(1, -1))
    finalPredict = ourModel.predict(X)
    toReturn = ""
    if finalPredict > 0.5:
        toReturn = "pure," + str(returnString[0])
    else:
        toReturn = "mixed," + str(returnString)
    return json.dumps(str(toReturn))

app.run(host='0.0.0.0')