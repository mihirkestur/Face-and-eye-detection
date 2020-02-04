import numpy as np
import cv2
fc=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#ec=cv2.CascadeClassifier('haarcascade_eye.xml') only for without glasses
eg=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
c=cv2.VideoCapture(0)
while True:
	_,f=c.read()
	gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
	faces=fc.detectMultiScale(gray,2,1)
	for (x,y,w,h) in faces:
		cv2.rectangle(f,(x,y),(x+w,y+h),(250,0,0),2)
		roi=gray[y:y+h,x:x+w]
		roic=f[y:y+h,x:x+w]
		eyes=eg.detectMultiScale(roi)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roic,(ex,ey),(ex+ew,ey+eh),(0,250,0),2)
	cv2.imshow('detect',f)
	if cv2.waitKey(1)==27:
		break
c.release()
cv2.destroyAllWindows()