import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model = load_model("/home/bunnys-weapon/Documents/mask_detector_model.h5")


def draw_label(img,text,pos,bg_color):
	text_size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
	
	end_x=pos[0]+text_size[0][0]+2
	end_y=pos[1]+text_size[0][1]+2

	cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
	cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

def detect_face_mask(img):
    img = img.reshape(1, 224, 224, 3)
    y_pred = model.predict(img)
    return 1 if y_pred[0][0] > 6.8215996e-07 else 0  

#with_mask--->0 without_mask--->1


haar=cv2.CascadeClassifier("/home/bunnys-weapon/Documents/haarcascade_frontalface_default.xml")

def detect_face(img):
	coods=haar.detectMultiScale(img)
	return coods


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (224, 224))
    y_pred = detect_face_mask(img)
    
    coods=detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    for x,y,w,h in coods:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    
    if y_pred==0:
    	draw_label(frame,"MASK",(30,30),(0,255,0))
    else:
    	draw_label(frame,"NO MASK",(30,30),(0,0,255))

    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

cap.release()  
cv2.destroyAllWindows()  



