import os
from platform import win32_edition
import cv2
import time
import numpy as np

file_path = input("파일 이름을 입력해주세요: ")
cap = cv2.VideoCapture('video/'+ file_path)
if not cap.isOpened():
    print("Could not Open :", 'video/' + file_path)
    exit(0)
print("file ok")

fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame/fps

t = 2000
target = t*fps
cap.set(cv2.CAP_PROP_POS_FRAMES,target)
ret, img = cap.read()
height, width, channel = img.shape
# print(height,width,channel)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_crop = img[0:int(height/4),0:int(width/4)]
height, width, channel = img_crop.shape
cv2.imshow("img",img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
img_gray_thr = cv2.adaptiveThreshold(
    img_gray,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)
cv2.imshow("img",img_gray_thr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img_blur = cv2.GaussianBlur(img_gray, ksize=(3,3), sigmaX=0)
# img_blur_thr = cv2.adaptiveThreshold(
#     img_blur,
#     maxValue=255,
#     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     thresholdType=cv2.THRESH_BINARY_INV,
#     blockSize=19,
#     C=9
# )
# cv2.imshow("img",img_blur_thr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours, _ = cv2.findContours(
    img_gray_thr,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)
img_contours = np.zeros((height, width, channel), dtype=np.uint8)
cv2.drawContours(img_contours, contours=contours, contourIdx=-1, color=(255,255,255))
cv2.imshow("img",img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_boundingbox = np.zeros((height, width, channel), dtype=np.uint8)
contours_dict = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img_boundingbox, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)
    
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })
cv2.imshow("img",img_boundingbox)
cv2.waitKey(0)
cv2.destroyAllWindows()

MIN_AREA, MAX_AREA = 0,2000
MIN_WIDTH, MIN_HEIGHT = 0, 0
MIN_RATIO, MAX_RATIO = 0, 10

possible_contours = []
cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['h'] / d['w']
    
    if MIN_AREA < area < MAX_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)
img_possiblebox = np.zeros((height, width, channel), dtype = np.uint8)
for d in possible_contours:
    cv2.rectangle(img_possiblebox, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

cv2.imshow("img",img_possiblebox)
cv2.waitKey(0)
cv2.destroyAllWindows() 