from PIL import Image
from pytesser import *
import numpy as np
import cv2

im=cv2.imread('fnord.tif')

text=image_to_string(Image.fromarray(np.array(im,dtype='uint8')),cleanup=True)
##image_file = 'im1.jpg'
##im = Image.open(image_file)
##text = image_to_string(im)
##text = image_file_to_string(image_file)
##text = image_file_to_string(image_file, graceful_errors=True)
print "=====output=======\n"
print text
