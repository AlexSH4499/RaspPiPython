import numpy as np
import cv2
from matplotlib import pyplot as plt

def process_image(name):
    img = cv2.imread(name)

    #Smooths out the image
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)

    #Plots the images side by side
    plot([img,dst])

#This is using a gradient vector and an angle
#Gradients always give a perpendicular vector to the level curve
def canny_gradient(name):
    img = cv2.imread(name,0)
    edges = cv2.Canny(img,100,200)
    
    plot([img,edges])


def plot(img_li):

    for i in img_li:
        plt.subplot(400)
        plt.imshow(i , cmap = 'gray')
        plt.title(str(i))
        plt.xticks([])
        plt.yticks([])


if __name__ == '__main__':
    process_image('random.jpg')
