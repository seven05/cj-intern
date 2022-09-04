import cv2
import os
import numpy as np

try:
    if not os.path.exists('./fill'):
        os.makedirs('./fill')
except OSError:
    print('error : creating dir fill')

for x in range(1500,6400):
    for i in range(0,4):
        if(os.path.isfile("number/%d_num_%d.jpg" %(x,i)) == False):
            continue
        img = cv2.imread('number\%d_num_%d.jpg' %(x,i), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        if(img[0][0] > 128 and img[h-1][0] > 128 and img[0][w-1] > 128 and img[h-1][w-1] > 128):
            img_thr = cv2.threshold(img, 128, 0, cv2.THRESH_BINARY)[1]
        else:    
            img_thr = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
        #img_edge = cv2.Canny(img_thr,100,600)
        '''
        contours = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        if(len(contours) == 0):
            continue
        big_contour = max(contours, key=cv2.contourArea)
        result = np.zeros_like(img)
        cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)
        '''
        cv2.imwrite('fill\%d_%d.jpg' %(x,i), img_thr)
        
exit()