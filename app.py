from flask import Flask, render_template, request
import os
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from PIL import Image
from matplotlib import pyplot as plt

app = Flask(__name__)

dic = {0: 'COVID19', 1: 'NORMAL', 2: 'PNEUMONIA', 3: 'TUBERCULOSIS'}

model = load_model(os.path.expanduser('~/Desktop/cifar_cnn.h5'))

model.make_predict_function()

def predict_label(img_path):
    image_test = cv2.imread(img_path)
    image_test = Image.fromarray(image_test, 'RGB')
    image_test = image_test.resize((256, 256))
    image_test = np.array(image_test)
    p = model.predict(np.expand_dims(image_test, 0))
    p = np.squeeze(p) #remove the wrapping; from (1,10) to (10)
    ind = np.argmax(p)
    return (dic[ind], p[ind])

@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
    return render_template("index1.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']
        
        img_path = 'static/' + img.filename
        img.save(img_path)
        
        p, acc = predict_label(img_path)
        
    return render_template("index1.html", prediction = p, accuracy = (acc*100), img_path=img_path)

if __name__ == '__main__':
    app.run(debug = True)