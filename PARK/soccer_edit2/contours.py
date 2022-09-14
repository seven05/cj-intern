from distutils.log import error
import cv2
import os
import time
import numpy as np

try:
    if not os.path.exists('./contours'):
        os.makedirs('./contours')
except OSError:
    print('error : creating dir contours')

for t in range(600,2000):
    img = cv2.imread('./test' + "/test_%d.jpg" %t, 1)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #contours, hierachy = cv2.findContours(img_gray, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #img_con = cv2.drawContours(img, contours, -1, (255,0,0), 1)
    #cv2.imwrite('./contours' + "/contours_%d.jpg" %t, img_con)
    contours = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if len(contours) == 0 :
        continue
    big_contour = max(contours, key=cv2.contourArea)
    result = np.zeros_like(img)
    result = cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)
    cv2.imwrite('./contours' + "/contours3_%d.jpg" %t, result)
'''
result_gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
contours2 = cv2.findContours(result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour2 = contours2[0]
epsilon = cv2.arcLength(contour2, True)
approx = cv2.approxPolyDP(contour2, epsilon, True)
result2 = np.zeros_like(img)
result2 = cv2.drawContours(result2, [approx], -1, (0,255,0), 3)
cv2.imwrite('./contours' + "/contours3_%d.jpg" %t, result2)
'''