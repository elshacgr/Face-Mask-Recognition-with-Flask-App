# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 12:00:42 2021

@author: ptbanknegaraindonesi
"""
import cv2
import numpy as np
from model_config import get_model
from flask import Flask, render_template, Response

model = get_model()

app = Flask(__name__)

from pygame import mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

results={0:'mask',1:'without mask'}
GR_dict={0:(0,255,0),1:(0,0,255)}
rect_size = 4
cap = cv2.VideoCapture(0) 
haarcascade = cv2.CascadeClassifier('C:/Users/ptbanknegaraindonesi/testweb/facemaskrecog/haarcascade_frontalface_default.xml')
def gen_frames():
    while True:
        success,im = cap.read()
        im=cv2.flip(im,1,1) 
    
        rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
        faces = haarcascade.detectMultiScale(rerect_size)
        for f in faces:
            (x, y, w, h) = [v * rect_size for v in f] 
        
            face_img = im[y:y+h, x:x+w]
            rerect_sized=cv2.resize(face_img,(224,224))
            normalized=rerect_sized/255.0
            reshaped=np.reshape(normalized,(1, 224,224,3))
            reshaped = np.vstack([reshaped])
            result=model.predict(reshaped)
        
            label=np.argmax(result,axis=1)[0]
      
            cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
            cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
            if(results[label] =='mask'):
                print("No Beep")
            elif(results[label] =='without mask'):
                sound.play()
                print("Beep") 
                
        ret, buffer = cv2.imencode('.jpg', im)
        im = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)