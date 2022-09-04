import cv2
import os
import numpy as np

try:
    if not os.path.exists('./resize'):
        os.makedirs('./resize')
except OSError:
    print('error : creating dir resize')

for x in range(1500,6400):
    for i in range(0,4):
        if(os.path.isfile('fill\%d_%d.jpg' %(x,i)) == False):
            continue
        img = cv2.imread('fill\%d_%d.jpg' %(x,i), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        img_resize = cv2.resize(img,dsize=(48,48))
        cv2.imwrite('resize\%d_%d.jpg' %(x,i), img_resize)
        
exit()