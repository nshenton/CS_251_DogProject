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
from flask import render_template
### end imports ###

#creat flask instance
app = Flask(__name__)

#load neural networks to predict pure vs. mixed
ourModel = load_model('bestModel.h5')
model = ResNet50()

#load pca from train dataset
pca = pickle.load( open( "pca.p", "rb" ) )

#define the location and the methods accepted
@app.route('/home', methods=['POST'])
def home():
    #receive the file
    data = request.files['fileToUpload']
    
    #convert to buffer stream
    b = BytesIO(data.read())
    
    #convert rgba to rgb
    im = Image.open(b).convert('RGB')
    
    #resize the image
    img = im.resize((224, 224), Image.ANTIALIAS)
    
    #prepreocess the input (to np array)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    listPreds = decode_predictions(preds, top=4)[0]
    predictionVector = np.array((listPreds[0][2],listPreds[1][2],listPreds[2][2],listPreds[3][2]))
    returnString = np.array((listPreds[0][1],listPreds[1][1],listPreds[2][1]))
    print(returnString)

    print(predictionVector)
    X = pca.transform(np.array([predictionVector]).reshape(1, -1))
    finalPredict = ourModel.predict(X)
    url0 = "https://ltrinity.w3.uvm.edu/cs148/animalsoundquiz/photos/" + str(returnString[0]) + ".jpg"
    url1 = "https://ltrinity.w3.uvm.edu/cs148/animalsoundquiz/photos/" + str(returnString[1]) + ".jpg"
    string0 = str(returnString[0]).replace('_',' ')
    string1 = str(returnString[1]).replace('_',' ')
    rendered = render_template('phpTemplatePure.html', \
            title = "My Generated Page", \
            breeds = [{"url":url0, "name": string0}])
    if finalPredict < 0.5:
            rendered = render_template('phpTemplateMixed.html', \
            title = "My Generated Page", \
            breeds = [{"url":url0, "name": string0},{"url":url1,"name": string1}])
    return(rendered)

app.run(host='0.0.0.0')