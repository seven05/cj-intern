from distutils.log import error
import cv2
import os
import time
import numpy as np
filepath = 'P470472958_EPI0001_01_t35.mp4'
cap = cv2.VideoCapture(filepath)
if not cap.isOpened():
    print("Could not Open :", filepath)
    exit(0)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame/fps
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
try:
    if not os.path.exists('./test'):
        os.makedirs('./test')
except OSError:
    print('error : creating dir test')

for t in range(600,2000):
    for i in range(0,5):
        target = (t-4+i*2)*fps
        cap.set(cv2.CAP_PROP_POS_FRAMES,target)
        globals()["ret{}".format(i)], globals()["img{}".format(i)] = cap.read()
        globals()["img{}_crop".format(i)] = globals()["img{}".format(i)][0:int(height/4),0:int(width/4)]
        globals()["img{}_gray".format(i)] = cv2.cvtColor(globals()["img{}_crop".format(i)], cv2.COLOR_BGR2GRAY)

    rows,cols = globals()["img2_gray"].shape
    temp_img = np.zeros((rows,cols))
    for i in range(rows):
            for j in range(cols):
                temp = abs(int(globals()["img2_gray"][i][j])-int(globals()["img0_gray"][i][j]))
                if(temp < 10):
                    temp = abs(int(globals()["img2_gray"][i][j])-int(globals()["img1_gray"][i][j]))
                    if(temp < 10):
                        temp = abs(int(globals()["img2_gray"][i][j])-int(globals()["img3_gray"][i][j]))
                        if(temp < 10):
                            temp = abs(int(globals()["img2_gray"][i][j])-int(globals()["img4_gray"][i][j]))
                            if(temp < 10):
                                temp_img[i][j] = 255
    #print(temp_img)           
    cv2.imwrite('./test' + "/test_%d.jpg" %(t), temp_img)

cap.release()
exit()