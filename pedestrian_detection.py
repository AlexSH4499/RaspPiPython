import argparse
import cv2
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np


'''Taken from PyImageSearch at:
    https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/'''

def pedestrian_detection():
    #initialize the HOG descriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    for imagePath in paths.list_images(args["images"]):
        #load the image and resize to reduce detection time
        # and improve accuracy
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=min(400,image.shape[1]))

        #remeber python must explicitly use copy()
        #Otherwise, it will just switch the pointer to the new variable
        orig = image.copy()

        #detection!
        (rects, weights) = hog.detectMultiscale(image, winStride=(4,4),\
                                                padding = (8,8), scale=1.05)

        #draw original bounding boxes
        for(x,y,w,h) in rects:
            cv2.rectangle(orig,(x,y),(x+w,y+h),(0,0,255),2)

        #Apply non-maxima suppression to bounding boxes
        rects = np.array([[x,y,x+w,y +h] for(x,y,w,h) in rects])
        pick = non_max_suppression(rects,probs=None, overlapThresh=0.65)

        for(xA, yA, xB, yB) in pick:
            cv2.rectangle(image,(xA,yA),(xB,yB),(0,255,0),2)

        filename = imagePath[imagePath.rfind("/")+1:]
        print("[INFO] {}: {} original boxes,{}after suppression".format(\
            filename,len(rects),len(pick)))

        cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)
        cv2.waitKey(0)
        
    
