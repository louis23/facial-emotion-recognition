#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os

# model2aug = tf.keras.models.load_model('./model_2_aug_nocall_BEST/model_2_aug_nocall_entire_model.h5')
model = tf.keras.models.load_model('./model/model_entire_model.h5')


# In[2]:


# get_ipython().system('pip install opencv-python')
import subprocess

# Install OpenCV using pip
subprocess.run(["pip", "install", "opencv-python"])



# 

# In[ ]:


##### Haarcascade

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown']
classifier = model

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Warning: Cannot open webcam. Make sure it's connected and not being used by another application.")

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            img = roi_gray.astype('float') / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)

            prediction = classifier.predict(img)[0]
            top_indices = np.argsort(prediction)[-2:]
            top_emotion = top_indices[1]
            label = emotions[top_emotion]
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('EMOTION DETECTOR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




