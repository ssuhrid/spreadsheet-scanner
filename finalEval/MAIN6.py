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
def order_points(pts):

    pts=np.array(pts,dtype="float32")
##    print pts
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
##    print diff
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
def four_point_transform(image, pts):
##    print pts
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
##    print rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

##    print maxWidth,maxHeight
    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

##    print dst
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
def findRegion(im):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    blur=cv2.GaussianBlur(gray,(1,1),0);
    show('blur',blur)
    cv2.waitKey(0)
    edges=cv2.Canny(blur,100,20)
    show('edges',edges)
    cv2.waitKey(0)

    cnts,ret=cv2.findContours(edges.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    screenCnt='False'
    for c in cnts:# loop over the contours
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    ##    print len(approx)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        print len(approx)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt != 'False':
        pts=screenCnt.flatten();

        pts=([pts[0],pts[1]],[pts[2],pts[3]],[pts[4],pts[5]],[pts[6],pts[7]])
        print pts
        # show the contour (outline) of the piece of paper
        print "STEP 2: Find contours of paper"
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", img)

        cv2.waitKey(0);
        
        warped = four_point_transform(img, pts)
        cv2.imshow('warped',warped)

        cv2.waitKey(0);








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

findRegion(img);

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
