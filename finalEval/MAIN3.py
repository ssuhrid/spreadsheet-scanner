import cv2
import numpy as np
from fpt import four_point_transform
import PyTesser.pytesser as pyt
import Image
import pyttsx

engine=pyttsx.init()
img=cv2.imread('Samples//priyank.jpg');

# img=cv2.resize(img.copy(),(0,0),fx=0.4,fy=0.4)
orig=img.copy()

lower_black = np.array([0,0,0])
upper_black = np.array([80,80,80])
black = cv2.inRange(img, lower_black, upper_black)
cv2.imshow('warped',black)
cv2.waitKey(0)
warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
text = pyt.image_to_string(Image.fromarray(np.array(black, dtype='uint8')), cleanup=True)
print "=====output=======\n"
print text
# text='my name is shamim akthar'
engine.say(text)
engine.runAndWait()
cv2.waitKey(0);

# thresh=cv2.resize(img.copy(),(0,0),fx=0.4,fy=0.4)

cv2.destroyAllWindows()
