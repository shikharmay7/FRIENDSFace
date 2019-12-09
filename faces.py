import cv2
import numpy as np 
import pickle

face_cascades = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')		#CascadeClassifier for face
eye_cascades = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')		#CascadeClassifier for eyes


recognizer = cv2.face.LBPHFaceRecognizer_create()	#if there is a problem here then, execute "pip install opencv-contrib-python" on the command line
recognizer.read("trainer.yml")


labels = {}

with open("labels.pickle",'rb') as f:		#open the file in read mode
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}	#inverting the dictionary

cap = cv2.VideoCapture(0)	#camera start

while(True):

	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascades.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5 )

	for (x,y,w,h) in faces:
		#print(x,y,w,h)
		
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]


		id_, conf = recognizer.predict(roi_gray)
		if conf>=45:
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 1
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		
		img_item = "my_image.png"
		img_itemcol = "my_colimg.png"
		
		cv2.imwrite(img_item, roi_gray)
		cv2.imwrite(img_itemcol, roi_color)


		color = (255, 0, 0)
		stroke = 2
		cv2.rectangle(frame, (x,y), (x+w, y+h), color, stroke)


		eyes = eye_cascades.detectMultiScale(roi_gray)	
	#cv2.imshow('gray',gray)
		for (ex,ey,ew,eh) in eyes:
			color = (0, 255, 0)
			stroke = 2
			cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), color, stroke)

	cv2.imshow('frame',frame)
	
	if(cv2.waitKey(20) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()
