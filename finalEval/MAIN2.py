import cv2
import numpy as np
from fpt import four_point_transform
import PyTesser.pytesser as pyt
import Image
import pyttsx

engine=pyttsx.init()
img=cv2.imread('Samples//suhrid.jpg');

# img=cv2.resize(img.copy(),(0,0),fx=0.4,fy=0.4)
orig=img.copy()

##cv2.imshow('img',img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
blur=cv2.GaussianBlur(gray,(21,21),0);
##blur=cv2.resize(blur.copy(),(0,0),fx=0.2,fy=0.2)
##cv2.imshow('blur',blur)
edges=cv2.Canny(blur,100,20)
##cv2.imshow('edges',edges)

##cv2.waitKey(0)
cv2.destroyAllWindows();

cnts,ret=cv2.findContours(edges.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

screenCnt='False'
# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
##    print peri
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
##    print len(approx)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    print len(approx)
    if len(approx) == 4:
        screenCnt = approx
        break
##print screenCnt

if screenCnt != 'False':
    pts=screenCnt.flatten();

##if len(pts)==8:
    pts=([pts[0],pts[1]],[pts[2],pts[3]],[pts[4],pts[5]],[pts[6],pts[7]])
    print pts
    # show the contour (outline) of the piece of paper
    print "STEP 2: Find contours of paper"
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", img)
    ##cv2.waitKey(0)
    cv2.destroyAllWindows()

    # apply the four point transform to obtain a top-down
    # view of the original image

    ##warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    warped = four_point_transform(orig, pts)
    cv2.imshow('warped',warped)
     
    white=np.ones((warped.shape),np.uint8)*255
##    print white
    warped=white-warped
    abc=cv2.resize(warped.copy(),(0,0),fx=0.2,fy=0.2)
    cv2.imshow('warped',abc)
    cv2.waitKey(0)
    ret,thresh=cv2.threshold(warped,220,255,cv2.THRESH_BINARY)
    thresh=cv2.resize(thresh.copy(),(0,0),fx=0.2,fy=0.2)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)









    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
##    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
##    thresh = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
##    ret,thresh=cv2.threshold(warped,240,255,cv2.THRESH_BINARY)
##    thresh=cv2.resize(thresh.copy(),(0,0),fx=0.4,fy=0.4)
##    cv2.imshow('ext',thresh)
##    cv2.imshow('thresh',thresh)
##    cv2.waitKey(0);
    ##cv2.adaptiveThreshold(warped,255,
    ##warped = threshold_adaptive(warped, 251, offset = 10)
    ##warped = warped.astype("uint8") * 255
    ##ret,warped=cv2.threshold(warped,100,255,cv2.THRESH_BINARY_INV)
    ##ret,warped=cv2.invert(warped)
     
    # show the original and scanned images
    print "STEP 3: Apply perspective transform"
    ##cv2.imshow('final',warped)
    ##cv2.imshow("Original", imutils.resize(orig, height = 650))
    ##cv2.imshow("Scanned", imutils.resize(warped, height = 650))
    ##cv2.imshow('abc',warped)

    # thresh=thresh[500:1000,:]

##    text=pyt.image_to_string(Image.fromarray(np.array(thresh,dtype='uint8')),cleanup=True)
    print "=====output=======\n"
    text = pyt.image_to_string(Image.fromarray(np.array(thresh, dtype='uint8')), cleanup=True)
    print text
    engine.say(text)
    engine.runAndWait()
    cv2.waitKey(0)
    cv2.destroyAllWindows();

else:
    lower_black = np.array([0,0,0])
    upper_black = np.array([80,80,80])
    black = cv2.inRange(img, lower_black, upper_black)
    abc=cv2.resize(black.copy(),(0,0),fx=0.2,fy=0.2)
    cv2.imshow('warped',abc)
    cv2.waitKey(0)
##    r,g,b=cv2.split(img)
##    boundaries = [(0, 0, 0], [50, 50, 50])
##    zeros=np.zeros((gray.shape),np.uint8)
##    for i,j in img:
##        if img[i][j][0]<bwthresh and img[i][j][1]<bwthresh and img[i][j][2]<bwthresh:
##            zeros[i][j]=255
    warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##    thresh = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
##    ret,warped=cv2.invert(warped)
##    warped=cv2.subtract(np.ones((warped.size),dtype='uint8')*255,warped)
####    white=np.ones((warped.shape),np.uint8)*255
######    print white
####    warped=white-warped
####    abc=cv2.resize(warped.copy(),(0,0),fx=0.2,fy=0.2)
####    cv2.imshow('warped',abc)
####    cv2.waitKey(0)
####    ret,thresh=cv2.threshold(warped,180,255,cv2.THRESH_BINARY)
####    thresh=cv2.resize(thresh.copy(),(0,0),fx=0.2,fy=0.2)
####    cv2.imshow('thresh',thresh)
####    cv2.waitKey(0)

    # text=pyt.image_to_string(pyt.Image.fromarray(np.array(abc,dtype='uint8')),cleanup=True)
    text = pyt.image_to_string(Image.fromarray(np.array(black, dtype='uint8')), cleanup=True)
    print "=====output=======\n"
    print 'No rectangular regions found'
    print text
    # text='my name is shamim akthar'
    engine.say(text)
    engine.runAndWait()
    cv2.waitKey(0);

    # thresh=cv2.resize(img.copy(),(0,0),fx=0.4,fy=0.4)

cv2.destroyAllWindows()
