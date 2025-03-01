from flask import Flask, render_template, request
import os
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from PIL import Image
from matplotlib import pyplot as plt

app = Flask(__name__)

dic = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'}

model = load_model(os.path.expanduser('~/Desktop/cifar_cnn.h5'))

model.make_predict_function()

def predict_label(img_path):
    image_test = cv2.imread(img_path)
    image_test = Image.fromarray(image_test, 'RGB')
    image_test = image_test.resize((64, 64))
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
        
        img_path = os.path.join('static', img.filename)
        
        print("Saving file to:", img_path)  # Debugging print
        
        if not os.path.exists('static'):
            print("Static folder is missing!")  # Debugging print
            os.mkdir('static')
        
        img.save(img_path)
        
        p, acc = predict_label(img_path)
        
    return render_template("index1.html", prediction = p, accuracy = (acc*100), img_path=img_path)

# @app.route("/submit", methods=['POST'])
# def get_hours():
#     if 'my_image' not in request.files:
#         return "No file part in request"

#     img = request.files['my_image']
#     if img.filename == "":
#         return "No file selected"

#     # Ensure 'static/' exists
#     static_dir = "static"
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)

#     img_path = os.path.join(static_dir, img.filename)

#     try:
#         img.save(img_path)
#         print(f"File saved to: {img_path}")  # Debugging
#         print(f"File exists? {os.path.exists(img_path)}")  # Debugging
#     except Exception as e:
#         print(f"Error saving file: {e}")

#     return render_template("index1.html", img_path=img_path)

if __name__ == '__main__':
    app.run(debug = True)