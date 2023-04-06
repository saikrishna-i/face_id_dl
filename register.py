import cv2
import mtcnn
import numpy as np
import models
import json
from keras import backend as K
import time

def add_to_face_details(name,vecs,face_details):
    avg = np.average(vecs,axis=0)
    vec_list = avg.tolist()
    face_details[name] = vec_list
    with open("face_details.json",'w') as fp:
        json.dump(face_details,fp,indent=2)
    

if __name__=='__main__':
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    input_shape = (3,225,225)
    # capture frames from a camera
    cap = cv2.VideoCapture(0)
    squeezenet = models.load_model()
    with open("face_details.json") as fp:
        face_details = json.load(fp)
    TIMER = 5
    vecs = []
    mode = ""

    # loop runs if capturing has been initialized.
    prev = time.time()

    while 1:
        ret, img = cap.read()

        # reads frames from a camera

        # convert to gray scale of each frames
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detects faces of different sizes in the input image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #faces = models.get_mtcnn_bboxes(img)
        face_vec = np.zeros((128))
        for (x,y,w,h) in faces:
            # To draw a rectangle in a face
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            
        
        if mode=='register':
            if TIMER >= 0:
    
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img, text=str(TIMER), 
                            org=(100, 100), fontFace=font,
                           fontScale= 3, color=(0, 255, 255),
                            thickness=4, lineType=cv2.LINE_AA)  

                # current time
                cur = time.time()
                face_small = models.extract_face(img, input_shape,(x,y,w,h))
                vecs.append(squeezenet.predict(face_small))

                
                if cur-prev >= 1:
                    prev = cur
                    TIMER = TIMER-1
            else :
                mode=""

        # Display an image in a window
        cv2.imshow('img',img)

        k = cv2.waitKey(30)
        if k == ord('r'):
            mode = 'register'
            prev = time.time()

            
        # Wait for Esc key to stop
        elif k == 27:
            break

    # Close the window
    cap.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()
    if TIMER==-1:
        name = input("Enter the name: ")
        add_to_face_details(name,vecs,face_details)