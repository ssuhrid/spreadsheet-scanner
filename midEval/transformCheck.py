import cv2
import numpy as np


cam=cv2.VideoCapture(0);
pts=[];

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

def set_coord(event,x,y,flags,param):
    global pts,img
    if event == cv2.EVENT_LBUTTONDOWN:
        print x,y
        if len(pts)<4:
            pts.append((x,y))
        else:
            print pts
##            order_points(np.array(pts,dtype="int32"))
            cutOut=four_point_transform(img,pts)
            cv2.imshow('cutOut',cutOut)
            cv2.waitKey(0)
            cv2.destroyWindow('cutOut')
            pts=[]

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


def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:    
        cv2.circle(img,(x,y),100,(255,0,0),-1)

img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',set_coord)
cv2.imshow('image',img)
vidFlag=False

while True:
    if vidFlag==True:
        ret,img=cam.read(0);
        img=cv2.flip(img,1);
        cv2.imshow('image',img)
    k=cv2.waitKey(1) & 0xFF;
    if k==32:
        vidFlag=not vidFlag;
            
    if(k == 27):
        break;

cam.release();
cv2.destroyAllWindows();
