import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd



#Contrast stretching
def contrastStretching(img,r1,r2,a,b,c):
    s1 = a*r1
    s2 = b*(r2-r1)+s1
    imgC = np.zeros((256,256), dtype=np.int32)
    for i in range(0,256):
        for j in range(0,256):
            r = img[i,j]
            if r<r1:
                imgC[i,j] = a*r
            elif r>r1 and r<r2:
                imgC[i,j] = b*(r-r1) +s1
            else: 
                imgC[i,j] = c*(r-r2) + s2

    imgC = imgC.astype(np.uint8)
    return imgC

def histeq(img):
    a = np.zeros((256,),dtype=np.float16)
    b = np.zeros((256,),dtype=np.float16)
    imghist = img

    height,width=img.shape

    #finding histogram
    for i in range(width):
        for j in range(height):
            g = imghist[j,i]
            a[g] = a[g]+1

    #performing histogram equalization
    tmp = 1.0/(height*width)
    b = np.zeros((256,),dtype=np.float16)

    for i in range(256):
        for j in range(i+1):
            b[i] += a[j] * tmp
        b[i] = round(b[i] * 255)

    # b now contains the equalized histogram
    b=b.astype(np.uint8)

    #Re-map values from equalized histogram into the image
    for i in range(width):
        for j in range(height):
            g = imghist[j,i]
            imghist[j,i]= b[g]

    imghist = imghist.astype(np.uint8)
    
    return imghist

def enhancement(img1,img2):
    imgadd = cv2.add(img1,img2)
    imgsub = cv2.subtract(img1, img2)
    imgfinal= cv2.add(imgadd, imgsub)
    return imgfinal

def thresholding(img,t):
    for i in range(0,256):
        for j in range(0,256):
            if img[i,j]>t:
                img[i,j] =0
            else:
                img[i,j] = 255
    return img

def dilation(img,mask):
    img = img.astype(np.float16)
    dilimg = np.zeros((256,256), dtype=np.float16)
    for i in range(1,255):
        for j in range(1,255):
            imgtemp = img[i-1:i+2, j-1:j+2]
            res = np.multiply(imgtemp,mask)
            dilimg[i,j] = np.amax(res)
    dilimg = dilimg.astype(np.uint8)
    return dilimg

def erosion(img,mask):
    img = img.astype(np.float16)
    eroimg = np.zeros((256,256), dtype=np.float16)
    for i in range(1,255):
        for j in range(1,255):
            imgtemp = img[i-1:i+2, j-1:j+2]
            res=[]
            for k in range(0,3):
                for m in range(0,3):
                    if mask[k][m] ==1:
                        a = imgtemp[k,m]
                        res.append(a)
            eroimg[i,j] = np.amin(res)
    eroimg = eroimg.astype(np.uint8)
    return eroimg

def edgeDetection(img):
    imgS = img.astype(np.float16)
    sobx=[[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]
    sobx = np.array(sobx, np.float16)
    soby =[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    soby = np.array(soby,np.float16)
    for i in range(1,254):
        for j in range(1,254):
            imgtemp = img[i-1:i+2, j-1:j+2]
            x = np.sum(np.multiply(sobx,imgtemp))
            y = np.sum(np.multiply(soby,imgtemp))
            pixvalue = np.sqrt(x**2 + y**2)
            imgS[i,j] = pixvalue
    imgS = imgS.astype(np.uint8)
    return imgS

def detectCircles(img,openedimg):
    imgcircle = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    detected_circles = cv2.HoughCircles(openedimg,  
                    cv2.HOUGH_GRADIENT, 10, minDist= 10, param2= 30, minRadius = 1, maxRadius = 13) 

    ctr = 0
    imgcirclefinal = imgcircle  # Initialize imgcirclefinal outside the if block

    if detected_circles is not None: 
        # Convert the circle parameters a, b, and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
        
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2]  # a, b are the coordinates of the center, and r is the radius

            # Draw the circumference of the circle. 
            imgcirclefinal = cv2.circle(imgcircle, (a, b), r, (0, 255, 0), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            # cv2.circle(img1, (a, b), 1, (255, 0, 0), 3) 
            ctr += 1 
    
    return imgcirclefinal, ctr

def detected_cell_counting(old_img):
    new_img = cv2.cvtColor(old_img, cv2.COLOR_BGR2HSV)

    # Set the thresholds for RBC Detection
    rlower = np.array([160, 60, 100])
    rupper = np.array([180, 200, 255])

    # Find the colors within the specified boundaries and apply the mask
    rmask = cv2.inRange(new_img, rlower, rupper)

    # Find the contours
    rcontours, hierarchy = cv2.findContours(rmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Count the number of RBC's
    rbc_count = 0
    rbc = []
    multiple_cells = []
    for contour in rcontours:
        a = cv2.contourArea(contour)
        if a > 75 and a <= 300:
            rbc.append(contour)
            rbc_count += 1
        elif a > 300:
            multiple_cells.append(contour)
            if a <= 600:
                rbc_count += 2
            else:
                rbc_count += 3

    # Print the count
    print(f'Approximate number of RBC: {rbc_count}')
    
    # Segment the RBC's
    cv2.drawContours(old_img, multiple_cells, -1, (0, 0, 255), 1)
    cv2.drawContours(old_img, rbc, -1, (0, 0, 255), 1)

    # Set the thresholds for WBC Detection
    wlower = np.array([120, 50, 20])
    wupper = np.array([140, 255, 255])

    # Find the colors within the specified boundaries and apply the mask
    wmask = cv2.inRange(new_img, wlower, wupper)

    # Find the contours
    wcontours, hierarchy = cv2.findContours(wmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Count and segment the WBC's
    cv2.drawContours(old_img, wcontours, -1, (255, 0, 0), 3)
    wbc_count = len(wcontours)

    # Print the count
    print(f'Approximate number of WBC: {wbc_count}')
    print(f'Total number of Cells: {rbc_count + wbc_count}')
    
    return rbc_count, wbc_count, new_img







