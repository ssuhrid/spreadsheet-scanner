import cv2
import numpy as np
# from fpt import four_point_transform
import PyTesser.pytesser as pyt
import Image
 # import pyttsx
#import time

def show(str, im):
    l=len(im.shape)
    if l==3:
        [h,w,x]=im.shape;
    else:
        [h,w,]=im.shape;
    # print h,w
    if h>480 or w>300:
        x=[480.0/h, 300.0/w];
        # print x
        x=max(x)
        if x<1.0:
            im=cv2.resize(im,(0,0),fx=x,fy=x)
    cv2.imshow(str,im);
def change(a):
    global thresh
    x=cv2.getTrackbarPos('Thresh','Image')
##    x=(x*2)+1;
##    blur = cv2.GaussianBlur(gray, (x, x), 0);
    lower_black = np.array([0,0,0])
    upper_black = np.array([x,x,x])
    thresh = cv2.inRange(img2, lower_black, upper_black)
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
##    cv2.imshow('warped',warped)
##    cv2.waitKey(0)
    return warped
def findRegion(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
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
        cv2.drawContours(im, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", im)
        cv2.waitKey(0);
        
        region = four_point_transform(im, pts)

        cv2.imshow('warped region',region)
        cv2.waitKey(0);
        
        cv2.destroyAllWindows();
        return region

# engine=pyttsx.init()

##cam=cv2.VideoCapture(1);
##time.sleep(1);
##while True:
##    ret,img=cam.read();
##    print img.shape
####    cv2.imshow('img',img);    
##    show('Camera',img);
##    if 0xFF & cv2.waitKey(1) == 27:
##        cv2.destroyWindow('Camera');
##        break;
##cam.release();
##img2=findRegion(img);

img=cv2.imread('S://Dropbox//PythonSuhrid//Major//midEval2//images//1.JPG');
# img2=img.copy()
show('Image',img)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
h,w=gray.shape;
ret,temp=cv2.threshold(gray,127,255,cv2.THRESH_BINARY);
# temp=abs(255-temp);
# show('Image',temp);

# kernel=np.ones((1,1),dtype=np.uint8);
# temp=cv2.erode(temp,kernel,iterations=1);
show('Image',temp);
cv2.waitKey(0);


# cv2.drawContours(img, contours, -1, (0,255,0), 2)
# show('Image',img);
# cv2.waitKey(0);
temp=abs(255-temp);
contours,hier = cv2.findContours(temp.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
maxIndex=0;maxArea=cv2.contourArea(contours[0]);
for i in range(0,len(contours)):
    r = cv2.boundingRect(contours[i]);
    area=r[2]*r[3];
    # print area
    if area>maxArea:
        maxIndex=i;
        maxArea=area;
maxPt=tuple(contours[maxIndex][0][0])
# print maxArea;
# print maxIndex
# print len(contours)
# # print maxPt
# cv2.circle(img,maxPt,1,(0,255,0),thickness=2,lineType=-1);
# # r=cv2.boundingRect(contours[maxIndex])
# cv2.drawContours(img,contours,maxIndex,(255,0,0),3);
# # rects=temp.copy()
# cv2.rectangle(img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),5)
# show('Image',img)
# zeros=np.zeros((gray.shape),dtype=np.uint8);
# cv2.drawContours(zeros,contours,maxIndex,127,3);
# print contours[maxIndex];

mask = np.zeros((h + 2, w + 2), np.uint8)
# Floodfill from point (0, 0)
cv2.floodFill(temp, mask, (tuple(contours[maxIndex][0][0])), 127);
show('Image',temp)
cv2.waitKey(0)

for i in range(0,h):
    for j in range(0,w):
        if temp[i][j]==127:
            temp[i][j]=255;
        else:
            temp[i][j]=0;
temp=abs(255-temp);
kernel=np.ones((3,3),dtype=np.uint8);
temp=cv2.erode(temp,kernel,iterations=1);

show('Image', temp)
cv2.waitKey(0)

# cv2.drawContours(img, contours, maxIndex, (0, 255, 0), 1);
# for i in range(0,len(contours[maxIndex])):
#     cv2.circle(temp,tuple(contours[maxIndex][i][0]),1,127,thickness=1,lineType=-1);
#     show('Image',temp);
#     cv2.waitKey(0);

contours,hier = cv2.findContours(temp.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
maxIndex=0;maxArea=cv2.contourArea(contours[0]);
for i in range(0,len(contours)):
    r = cv2.boundingRect(contours[i]);
    area=r[2]*r[3];
    # print area
    if area>maxArea:
        maxIndex=i;
        maxArea=area;
for i in range(0,len(contours)):
    r=cv2.boundingRect(contours[len(contours)-1-i])
    # r=cv2.boundingRect(contours[i]);
    rects=temp.copy()
    # img2=img.copy()
    # cv2.drawContours(img2,contours,maxIndex,(0,255,0),3);
    if i!=len(contours)-1-maxIndex:
        cv2.rectangle(rects,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),127,5)
        show('Image',rects);
        cv2.waitKey(0);

# cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0);
