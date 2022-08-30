import cv2
import os
import numpy as np

try:
    if not os.path.exists('./fill2'):
        os.makedirs('./fill2')
except OSError:
    print('error : creating dir fill2')

x = 2100
img = cv2.imread('Crop2\%d_1.jpg' %x, cv2.IMREAD_GRAYSCALE)
hh, ww = img.shape[:2]
thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)
result = np.zeros_like(img)
cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)
cv2.imwrite('fill2\%d_1.jpg' %x, result)
exit()