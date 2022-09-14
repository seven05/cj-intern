from distutils.log import error
import cv2
import os
import time
filepath = 'P470472958_EPI0001_01_t35.mp4'
cap = cv2.VideoCapture(filepath)
if not cap.isOpened():
    print("Could not Open :", filepath)
    exit(0)
#cap.set(cv2.CAP_PROP_POS_FRAMES,20000)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame/fps
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

try:
    if not os.path.exists('./Captures'):
        os.makedirs('./Captures')
except OSError:
    print('error : creating dir Captures')

start = time.time()
count = 0
while(cap.get(cv2.CAP_PROP_POS_FRAMES) <= frame-100):
    ret, img = cap.read()
    if(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % fps == 0):
        #img = cv2.resize(img,dsize=(1280,720))
        img_crop = img[0:int(height/2),0:int(width/2)]
        #edges = cv2.Canny(img_crop, 100,600)
        #cv2.imwrite('./Captures' + "/edges_%d.jpg" %count, edges)
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        ret, img_thr = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
        #cv2.imwrite('./Captures' + "/thresh_%d.jpg" %count, img_thr)
        contours, hierachy = cv2.findContours(img_thr, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img_con = cv2.drawContours(img_crop, contours, -1, (255,0,0), 2)
        cv2.imwrite('./Captures' + "/contours_%d.jpg" %count, img_con)
        count +=1
print("time :", time.time() - start)

cap.release()
exit()