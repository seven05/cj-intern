from distutils.log import error
import cv2
import os
import time
filepath = 'P470472958_EPI0150_01_t35.mp4'
cap = cv2.VideoCapture(filepath)
if not cap.isOpened():
    print("Could not Open :", filepath)
    exit(0)
#cap.set(cv2.CAP_PROP_POS_FRAMES,20000)
#cap.set(cv2.CAP_PROP_POS_FRAMES,24000)
#cap.set(cv2.CAP_PROP_POS_FRAMES,104760)
#cap.set(cv2.CAP_PROP_POS_FRAMES,106700)
'''
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
'''
fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame/fps
#print(duration)
'''
print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)
'''
'''
ret, img = cap.read()
img = cv2.resize(img,dsize=(1280,720))
img_crop = img[35:65,80:140]
'''
'''
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
cv2.imshow('img_crop',img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
edges = cv2.Canny(img, 550, 700)
cv2.imshow('edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
edges = cv2.Canny(img_crop, 100,600)
cv2.imshow('edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
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
        img = cv2.resize(img,dsize=(1280,720))
        img_crop = img[35:65,80:140]
        edges = cv2.Canny(img_crop, 100,600)
        cv2.imwrite('./Captures' + "/%d.jpg" %count, edges)
        count +=1
print("resize time :", time.time() - start)
cap.release()

cap = cv2.VideoCapture(filepath)
if not cap.isOpened():
    print("Could not Open :", filepath)
    exit(0)
'''
try:
    if not os.path.exists('./Captures2'):
        os.makedirs('./Captures2')
except OSError:
    print('error : creating dir Captures2')

start = time.time()
count = 0
while(cap.get(cv2.CAP_PROP_POS_FRAMES) <= frame-100):
    ret, img = cap.read()
    if(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % fps == 0):
        #img = cv2.resize(img,dsize=(1280,720))
        img_crop = img[52:98,120:210]
        edges = cv2.Canny(img_crop, 100,600)
        cv2.imwrite('./Captures2' + "/%d.jpg" %count, edges)
        count +=1
print("time :", time.time() - start)

cap.release()
exit()
