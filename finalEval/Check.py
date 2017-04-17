import cv2
import numpy as np

def show(str, im):
    [h,w,]=im.shape
    if h>480 or w>300:
        x=[480.0/h, 300.0/w];
        print x
        x=max(x)
        im=cv2.resize(im,(0,0),fx=x,fy=x)
    cv2.imshow(str,im);
    # cv2.waitKey(0)

def change(a):
    global gray
    x=cv2.getTrackbarPos('Blur','Image')
    x=(x*2)+1;
    blur = cv2.GaussianBlur(gray, (x, x), 0);

    c1=cv2.getTrackbarPos('Canny 1','Image')
    c2=cv2.getTrackbarPos('Canny 2','Image')

    edges = cv2.Canny(blur, c1, c2)
    show('Image',edges)
    pass

if __name__== "__main__":
    img=cv2.imread('Samples//im4.jpg');
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    show('Image',gray)
    cv2.createTrackbar('Blur', 'Image', 1, 20, change);
    cv2.createTrackbar('Canny 1', 'Image', 1, 100, change);
    cv2.createTrackbar('Canny 2', 'Image', 1, 100, change);

    cv2.waitKey(0)
    cv2.destroyAllWindows();


    # cv2.imshow('Image',gray)
    # cv2.waitKey(0)

    # show('edges',edges)


    cv2.waitKey(0)
    cv2.destroyAllWindows();
