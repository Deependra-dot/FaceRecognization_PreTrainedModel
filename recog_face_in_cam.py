# -*- coding: utf-8 -*-
import cv2
from recognize_face import recognize_face

def recognize_faces_in_cam(input_embeddings, model):
    

    cv2.namedWindow("Face Recognizer")
    vc = cv2.VideoCapture(0)
   

    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        height, width, channels = frame.shape

        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through all the faces detected 
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h

           
            
            face_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]    
            identity, confidence = recognize_face(face_image, input_embeddings, model)
            
            

            if identity is not None:
                img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,255,255),2)
                cv2.putText(img, str(identity), (x1+5,y1-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x2+5,y2-5), font, 1, (255,255,255), 2)

        key = cv2.waitKey(100)
        cv2.imshow("Face Recognizer", img)

        if key == 27: # exit on ESC
            break
    vc.release()
    cv2.destroyAllWindows()
