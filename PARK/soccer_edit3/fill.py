import cv2
import os
import numpy as np

try:
    if not os.path.exists('./fill'):
        os.makedirs('./fill')
except OSError:
    print('error : creating dir fill')

def edge_cut(img):
        status = 0
        x_min,y_min,x_max,y_max = 0,0,0,0
        cols, rows = img.shape
        for x in range(0,rows):
            for y in range(0,cols):
                if (img[y][x] == 255):
                    x_min = x
                    status = 1
                    break
                if (status == 1):
                    break
        status = 0

        for y in reversed(range(cols)):
            for x in reversed(range(rows)):
                if (img[y][x] == 255):
                    x_max = x
                    y_max = y
                    status = 1
                    break
                if (status == 1):
                    break
        status = 0

        for y in range(0,cols):
            for x in range(0,rows):
                if (img[y][x] == 255):
                    y_min = y
                    status = 1
                    break
                if (status == 1):
                    break
        status = 0

        for x in reversed(range(rows)):
            for y in reversed(range(cols)):
                if (img[y][x] == 255):
                    if (y >= y_max):
                        y_max = y
                    if (x >= x_max):
                        x_max = x
                    status = 1
                    break
                if (status == 1):
                    break
        status = 0
        return x_min,y_min,x_max,y_max

for n in range(2020,2021):
    for i in range(0,4):
        if(os.path.isfile("number/%d_num_%d.jpg" %(n,i)) == False):
            continue
        img = cv2.imread('number/%d_num_%d.jpg' %(n,i), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        if(img[0][0] > 128 and img[h-1][0] > 128 and img[0][w-1] > 128 and img[h-1][w-1] > 128):
            img_thr = cv2.threshold(img, 128, 0, cv2.THRESH_BINARY)[1]
        else:
            img_thr = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

        for j in range(0,w):
            print(img_thr[0][j])

        x_min,y_min,x_max,y_max = edge_cut(img_thr)
        print(x_min,y_min,x_max,y_max)

        if(x_min == x_max or y_min == y_max):
            continue
        img_edge_cut = img_thr[y_min:y_max,x_min:x_max]
        cv2.imwrite('fill/%d_%d.jpg' %(x,i), img_edge_cut)       
exit()