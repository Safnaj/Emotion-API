import base64
import numpy as np
import io
import cv2
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS
from io import BytesIO


app = Flask(__name__)
CORS(app)

#Load Model Function
def get_model():
  global model

  model = model_from_json(open("Emotions.json", "r").read())
  model.load_weights('Emotions.h5')
  print("Model Loaded..!")

get_model()


#Image Preprocessing
def preprocess_image(data):
  
  nparr = np.fromstring(base64.b64decode(data), np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

  for (x,y,w,h) in faces_detected:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray = gray_img[y:y+w,x:x+h] #cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        return img_pixels


@app.route('/')
def hello_world():
    return 'Welcome to Emotion Detection API..!'


@app.route('/predict', methods=['POST', 'GET'])
def predict():
  message = request.get_json(force=True)
  encoded = message['image']
  
  processed_image = preprocess_image(encoded)
  predictions = model.predict(processed_image)

  max_index = np.argmax(predictions[0])
  emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')      
  predicted_emotion = emotions[max_index]

  print(predicted_emotion)
  return jsonify(predicted_emotion)
   

app.run(host='0.0.0.0', port=8000)