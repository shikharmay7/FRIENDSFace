import cv2
import os
import numpy as np
from PIL import Image
import pickle


base_dir = os.path.dirname(os.path.abspath(__file__))
#print(base_dir)
image_dir = os.path.join(base_dir,"images")
#print(image_dir)

face_cascades = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	#print("root ", root)
	#print("dirs ", dirs)
	#print("files ",files)
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			#print(path)
			#print(root)
			#print(os.path.basename(root))
			label = os.path.basename(root).lower()
			#print(label,path)

			if label not in label_ids:
				label_ids[label] = current_id
				current_id+=1
			
			id_ = label_ids[label]

			#y_labels.append(label)
			#x_train.append(path)

			pil_image = Image.open(path).convert("L")	#grayscale
			
			size = (550,550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image,"uint8")	#convert into np array
			#print(image_array)

			faces = face_cascades.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5 )

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

#print(label_ids)
#print(y_labels)
#print(x_train)

with open("labels.pickle",'wb') as f:
	pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")