import cv2
import numpy as np
from fpt import four_point_transform
import PyTesser.pytesser as pyt
import Image
import pyttsx

def show(str, im):
    l=len(im.shape)
    if l==3:
        [h,w,x]=im.shape;
    else:
        [h,w,]=im.shape;
    if h>480 or w>300:
        x=[480.0/h, 300.0/w];
        print x
        x=max(x)
        im=cv2.resize(im,(0,0),fx=x,fy=x)
    cv2.imshow(str,im);

def change(a):
    global thresh
    x=cv2.getTrackbarPos('Thresh','Image')
##    x=(x*2)+1;
##    blur = cv2.GaussianBlur(gray, (x, x), 0);
    lower_black = np.array([0,0,0])
    upper_black = np.array([x,x,x])
    thresh = cv2.inRange(img, lower_black, upper_black)
    show('Image',thresh)
    pass


engine=pyttsx.init()
img=cv2.imread('Samples//priyank.jpg');

cam=cv2.VideoCapture(1);
while True:
    ret,img=cam.read();
    show('Camera',img);
    if 0xFF & cv2.waitKey(1) == 27:
        cv2.destroyWindow('Camera');
        break;
cam.release();

show('Image',img)
cv2.waitKey(0);
x1=120;
lower_black = np.array([0,0,0])
upper_black = np.array([x1,x1,x1])
thresh = cv2.inRange(img, lower_black, upper_black)
show('Image',thresh)
cv2.createTrackbar('Thresh', 'Image', x1, 255, change);

cv2.waitKey(0);
##cv2.destroyAllWindows();

orig=img.copy()

##cv2.imshow('warped',black)
##show('Image',black);
cv2.waitKey(0);
####warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
text = pyt.image_to_string(Image.fromarray(np.array(thresh, dtype='uint8')), cleanup=True)
print "=====output=======\n"
print text
engine.say(text)
engine.runAndWait()
cv2.waitKey(0);

# thresh=cv2.resize(img.copy(),(0,0),fx=0.4,fy=0.4)

cv2.destroyAllWindows()
