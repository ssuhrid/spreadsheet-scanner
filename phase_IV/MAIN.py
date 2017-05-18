import cv2
import numpy as np
# from fpt import four_point_transform
import PyTesser.pytesser as pyt
from PIL import Image
import requests
# import time

def show(str, im):
    H,W=(480.0,300.0)
##    H,W=(480.0*0.8,300.0*0.8)
    l = len(im.shape)
    if l == 3:
        [h, w, x] = im.shape;
    else:
        [h, w, ] = im.shape;
    # print h,w
    if h > H or w > W:
        x = [H / h, W / w];
        # print x
        x = max(x)
        if x < 1.0:
            im = cv2.resize(im, (0, 0), fx=x, fy=x)
    cv2.imshow(str, im);
def change(a):
    global thresh
    x = cv2.getTrackbarPos('Thresh', 'Image')
    ##    x=(x*2)+1;
    ##    blur = cv2.GaussianBlur(gray, (x, x), 0);
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([x, x, x])
    thresh = cv2.inRange(img2, lower_black, upper_black)
    show('Image', thresh)
    pass
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    ##    print pts
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
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
        [0, maxHeight - 1]], dtype="float32")

    ##    print dst
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    ##    cv2.imshow('warped',warped)
    ##    cv2.waitKey(0)
    return warped
def findRegion(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    blur = cv2.GaussianBlur(gray, (1, 1), 0);
##    show('blur', blur)
##    cv2.waitKey(0)
    edges = cv2.Canny(blur, 100, 20)
##    show('edges', edges)
##    cv2.waitKey(0)

    cnts, ret = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = 'False'
    for c in cnts:  # loop over the contours
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
        pts = screenCnt.flatten();

        pts = ([pts[0], pts[1]], [pts[2], pts[3]], [pts[4], pts[5]], [pts[6], pts[7]])
        print pts
        # show the contour (outline) of the piece of paper
        print "STEP 2: Find contours of paper"
        cv2.drawContours(im, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", im)
        cv2.waitKey(0);

        region = four_point_transform(im, pts)

        cv2.imshow('warped region', region)
        cv2.waitKey(0);

        cv2.destroyAllWindows();
        return region
def findLargestContourIndex(contours):
    maxIndex = 0;
    maxArea = cv2.contourArea(contours[0]);
    for i in range(0, len(contours)):
        r = cv2.boundingRect(contours[i]);
        area = r[2] * r[3];
        # print area
        if area > maxArea:
            maxIndex = i;
            maxArea = area;
    # print maxArea;
    # print maxIndex
    # print len(contours)
    return maxIndex;
def showEachCell(im,imGrid):
    contours, hier = cv2.findContours(imGrid.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = arrangeCells(imGrid,contours)
    for i in range(0, len(contours)):
        r = cv2.boundingRect(contours[i])
        # cv2.drawContours(img2,contours,i,(0,255,0),3);
        rects = im.copy()
        cv2.rectangle(rects, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0,255,0), 10)
        roi = im[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
        checkPresent(roi)
        show('Image',rects)
        cv2.waitKey(0)
def findAttendance(im,imGrid,days,noOfStudents):
    contours, hier = cv2.findContours(imGrid.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = arrangeCells(imGrid,contours)
    studentDetails = [];
    studentAttend = np.zeros(noOfStudents, dtype=np.uint8);
    boundFlag = False #to check for outer boundary
    for I,c in enumerate(contours):
        # print i
        r = cv2.boundingRect(c)
        roi = im[r[1]:r[1] + r[3], r[0]:r[0] + r[2]];
##        print imGrid.size
##        print r[2]*r[3]
        if imGrid.size / (r[2]*r[3]) < 2:
            boundFlag = True
            continue
        rects = img.copy()
##        img2=img.copy()
##        show('Image',img2)
        cv2.rectangle(rects, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), 127, 5)
##        show('Image',rects)
##        cv2.waitKey(0)
        # show('Image', rects);
        # show('ROI', roi );
        if boundFlag == True:
            i=I-1
        else:
            i=I
        if i % days == 0: #Serial Number
            pass;
        elif i % days == 1:
##            show('Image', roi);
##            cv2.waitKey(0);
            roi = cv2.resize(roi, (0, 0), fx=2, fy=2)
##            text='student %d'%(i/days)
            text = pyt.image_to_string(Image.fromarray(np.array(roi, dtype='uint8')), cleanup=True)
            text = text.lstrip();
            text = text.rstrip();
            studentDetails.append(text);
##            print text,
##            print ': ',
        else:
            if checkPresent(roi):
                studentAttend[i/days]+=1

    return studentDetails, studentAttend
def getLargestContour(im):
    contours, hier = cv2.findContours(im.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    maxIndex = findLargestContourIndex(contours);
    maxPt = tuple(contours[maxIndex][0][0])
    # # print maxPt
    # cv2.circle(img,maxPt,1,(0,255,0),thickness=2,lineType=-1);
    h,w=im.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im, mask, maxPt, 127);
    # show('Image', im)
    # cv2.waitKey(0)

    # for i in range(0, h):
    #     for j in range(0, w):
    #         if temp[i][j] == 127:
    #             temp[i][j] = 255;
    #         else:
    #             temp[i][j] = 0;
    im = cv2.inRange(im, 127, 127);
    return im;
# engine=pyttsx.init()

def findGrid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    h, w = gray.shape;
    ret, temp = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY);
##    show('Image', gray)
##    cv2.waitKey(0);

    # kernel=np.ones((1,1),dtype=np.uint8);
    # temp=cv2.erode(temp,kernel,iterations=1);
    # show('Image', temp);
    # cv2.waitKey(0);
    # cv2.drawContours(img, contours, -1, (0,255,0), 2)
    # show('Image',img);
    # cv2.waitKey(0);

    temp = abs(255 - temp);

    imGrid = getLargestContour(temp);

    imGrid = abs(255 - imGrid);

##    show('Image',imGrid)
##    cv2.waitKey(0);

    kernel = np.ones((25, 25), dtype=np.uint8);
    imGrid = cv2.erode(imGrid, kernel, iterations=1);
##    mask = np.zeros((h + 2, w + 2), np.uint8)
##    cv2.floodFill(imGrid,mask,(0,0),0);

    return imGrid

def checkPresent(roi):
    white=cv2.inRange(roi,(200,200,200),(255,255,255))
    white = 255 - white
    white = cv2.divide(white,255*np.ones((white.shape),dtype=np.uint8))
    roi=cv2.multiply(roi.copy(),cv2.merge([white,white,white]))
    b,g,r=cv2.split(roi)
    r=cv2.inRange(r,200,255)
    kernel = np.ones((5, 5), dtype=np.uint8);
    r = cv2.erode(r, kernel, iterations=1);
    r = cv2.dilate(r, kernel, iterations=1);
    avg=float(sum(sum(r)))/r.size

##    print avg
    if avg<0.5:
##        print 'P'
        return True
    else:
##        print 'A'
        return False

def arrangeCells(im,contours):
    centroids = []
    Ky=im.shape[0]/10
    Kx=37
    for c in contours:
        r=cv2.boundingRect(c)
        pt=(r[0]+r[2]/2,r[1]+r[3]/2)
        centroids.append(pt)
    for i in range(0,len(centroids)):
        for j in range(0,len(centroids)-1):
            if centroids[j][1]/Ky > centroids[j+1][1]/Ky :
               temp=centroids[j]
               centroids[j]=centroids[j+1]
               centroids[j+1]=temp
               temp=contours[j]
               contours[j]=contours[j+1]
               contours[j+1]=temp
    for i in range(0,len(centroids)):
        for j in range(0,len(centroids)-1):
            if centroids[j][0]/Kx>centroids[j+1][0]/Kx and centroids[j][1]/Ky==centroids[j+1][1]/Ky :
               temp=centroids[j]
               centroids[j]=centroids[j+1]
               centroids[j+1]=temp
               temp=contours[j]
               contours[j]=contours[j+1]
               contours[j+1]=temp
    return contours

def simplify(im,imGrid):

    imGrid = 255-imGrid
    contours,ret=cv2.findContours(imGrid.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    peri = cv2.arcLength(contours[0],True)
    approx = cv2.approxPolyDP(contours[0], 0.1 * peri, True)
    pts=approx.flatten()
##    cv2.rectangle(imGrid,(pts[0],pts[1]),(pts[4],pts[5]),127,100)
    print pts
    pts=([pts[0],pts[1]],[pts[2],pts[3]],[pts[4],pts[5]],[pts[6],pts[7]])

##    show('Image', imGrid)
##    cv2.waitKey(0);

    imGrid = four_point_transform(imGrid.copy(), pts)
    im = four_point_transform(im.copy(), pts)

##    show('Image', imGrid)
##    cv2.waitKey(0);

    return im,imGrid

if __name__ == '__main__':

    cv2.destroyAllWindows()

    days=7;noOfStudents=10;
    img = cv2.imread('images//cam01.JPG');
##    show('Image', img)
##    cv2.waitKey(0);

    imGrid=findGrid(img)

    img,imGrid = simplify(img,imGrid)
    
##    show('Image', img)
##    cv2.waitKey(0);

    days=days+2;

##    showEachCell(img,imGrid) #select and show each cell
    studentDetails,studentAttend= findAttendance(img,imGrid,days,noOfStudents); #find Attendance
##
    days-=2
    for i in range(0,len(studentDetails)):
##        print studentDetails[i],
##        print float(studentAttend[i]*100/days),
##        print '%'

        split = studentDetails[i].split(' ', 1)
        enroll = int(split[0])
        name = split[1]
        print enroll,
        print name,
        print studentAttend[i],
        data = {'enroll':enroll,'name':name,'attend':studentAttend[i]}
        r = requests.post('http://ssuhrid.com/updateDB.php', params=data)
        print r.status_code

    show('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ##exit(0);

