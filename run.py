# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2
import mtcnn
import numpy as np
import models
import json
from keras import backend as K




def get_name(face_vec,face_details,thresh = 0.05):
    min_dist = np.inf
    min_name = ""

    for name, vec_list in face_details.items():
        vec = np.array(vec_list)
        dist = models.euclidean_distance((face_vec,vec))
        print(dist)
        if dist<min_dist:
            min_name = name
            min_dist = dist
    print(dist)
    if min_dist>thresh:
        return "unkown"
    else:
        return min_name





if __name__=='__main__':
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    input_shape = (3,225,225)
    # capture frames from a camera
    cap = cv2.VideoCapture(0)
    squeezenet = models.load_model()
    with open("face_details.json") as fp:
        face_details = json.load(fp)
    # loop runs if capturing has been initialized.
    while 1:

        # reads frames from a camera
        ret, img = cap.read()

        # convert to gray scale of each frames
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detects faces of different sizes in the input image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #faces = get_mtcnn_bboxes()
        for (x,y,w,h) in faces:
            # To draw a rectangle in a face
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            face_small = models.extract_face(img, input_shape,(x,y,w,h))
            face_vec = squeezenet.predict(face_small)
            name = get_name(face_vec,face_details)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img=img, text=str(name), 
                            org=(x, y), fontFace=font,
                           fontScale= 0.75, color=(0, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)  
            
        # Display an image in a window
        cv2.imshow('img',img)

        # Wait for Esc key to stop
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Close the window
    cap.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()