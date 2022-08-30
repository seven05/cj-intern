from distutils.log import error
from scipy import stats
import cv2
import os
import time
import numpy as np
import datetime

def histogram(img):
    #print(filepath)
    rows,cols = img.shape
    x_hist = np.zeros(rows)
    y_hist = np.zeros(cols)
    hist = np.zeros(rows+cols)
    for i in range(rows):
        count = 0
        for j in range(cols):
            if(img[i,j] > 128):
                count += 1
        x_hist[i] += count
    for i in range(cols):
        count = 0
        for j in range(rows):
            if(img[j,i] > 128):
                count += 1
        y_hist[i] += count
    hist = np.concatenate((x_hist,y_hist))
    return hist

hist_min1_num0 = histogram(cv2.imread("number\min1/0.jpg",cv2.IMREAD_GRAYSCALE))
hist_min1_num1 = histogram(cv2.imread("number\min1/1.jpg",cv2.IMREAD_GRAYSCALE))
hist_min1_num2 = histogram(cv2.imread("number\min1/2.jpg",cv2.IMREAD_GRAYSCALE))
hist_min1_num3 = histogram(cv2.imread("number\min1/3.jpg",cv2.IMREAD_GRAYSCALE))
hist_min1_num4 = histogram(cv2.imread("number\min1/4.jpg",cv2.IMREAD_GRAYSCALE))
hist_min1_num5 = histogram(cv2.imread("number\min1/5.jpg",cv2.IMREAD_GRAYSCALE))
hist_min1_num6 = histogram(cv2.imread("number\min1/6.jpg",cv2.IMREAD_GRAYSCALE))
hist_min1_num7 = histogram(cv2.imread("number\min1/7.jpg",cv2.IMREAD_GRAYSCALE))
hist_min1_num8 = histogram(cv2.imread("number\min1/8.jpg",cv2.IMREAD_GRAYSCALE))
hist_min1_num9 = histogram(cv2.imread("number\min1/9.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num0 = histogram(cv2.imread("number\min2/0.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num1 = histogram(cv2.imread("number\min2/1.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num2 = histogram(cv2.imread("number\min2/2.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num3 = histogram(cv2.imread("number\min2/3.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num4 = histogram(cv2.imread("number\min2/4.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num5 = histogram(cv2.imread("number\min2/5.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num6 = histogram(cv2.imread("number\min2/6.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num7 = histogram(cv2.imread("number\min2/7.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num8 = histogram(cv2.imread("number\min2/8.jpg",cv2.IMREAD_GRAYSCALE))
hist_min2_num9 = histogram(cv2.imread("number\min2/9.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec1_num0 = histogram(cv2.imread("number\sec1/0.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec1_num1 = histogram(cv2.imread("number\sec1/1.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec1_num2 = histogram(cv2.imread("number\sec1/2.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec1_num3 = histogram(cv2.imread("number\sec1/3.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec1_num4 = histogram(cv2.imread("number\sec1/4.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec1_num5 = histogram(cv2.imread("number\sec1/5.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num0 = histogram(cv2.imread("number\sec2/0.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num1 = histogram(cv2.imread("number\sec2/1.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num2 = histogram(cv2.imread("number\sec2/2.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num3 = histogram(cv2.imread("number\sec2/3.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num4 = histogram(cv2.imread("number\sec2/4.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num5 = histogram(cv2.imread("number\sec2/5.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num6 = histogram(cv2.imread("number\sec2/6.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num7 = histogram(cv2.imread("number\sec2/7.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num8 = histogram(cv2.imread("number\sec2/8.jpg",cv2.IMREAD_GRAYSCALE))
hist_sec2_num9 = histogram(cv2.imread("number\sec2/9.jpg",cv2.IMREAD_GRAYSCALE))

def num_detection(target):
    state = True
    board_num = []
    cap.set(cv2.CAP_PROP_POS_FRAMES,target)
    ret, img = cap.read()
    img_crop = img[52:98,120:210]
    img_edge = cv2.Canny(img_crop, 150, 600)
    crop_num1 = img_edge[11:33,5:22]
    crop_num2 = img_edge[11:33,23:40]
    crop_num3 = img_edge[11:33,50:67]
    crop_num4 = img_edge[11:33,68:85]
    min1_hist = histogram(crop_num1)
    min2_hist = histogram(crop_num2)
    sec1_hist = histogram(crop_num3)
    sec2_hist = histogram(crop_num4)
    hist_gap1 = []
    hist_gap2 = []
    hist_gap3 = []
    hist_gap4 = []
    for i in range(0,10):
        temp1 = [abs(globals()[f'hist_min1_num{i}'][j] - min1_hist[j]) for j in range(len(min1_hist))]
        temp2 = [abs(globals()[f'hist_min2_num{i}'][j] - min2_hist[j]) for j in range(len(min2_hist))]
        temp4 = [abs(globals()[f'hist_sec2_num{i}'][j] - sec2_hist[j]) for j in range(len(sec1_hist))]
        hist_gap1.append(sum(temp1))
        hist_gap2.append(sum(temp2))
        hist_gap4.append(sum(temp4))
        temp1 = []
        temp2 = []
        temp4 = []
    for i in range (0,6):
        temp3 = [abs(globals()[f'hist_sec1_num{i}'][j] - sec1_hist[j]) for j in range(len(sec1_hist))]
        hist_gap3.append(sum(temp3))
        temp3 = []
    min1 = np.argmin(hist_gap1)
    min2 = np.argmin(hist_gap2)
    sec1 = np.argmin(hist_gap3)
    sec2 = np.argmin(hist_gap4)
    #print(hist_gap)
    if(hist_gap1[min1] < 70):
        board_num.append(min1)
    else:
        board_num.append(10)
    if(hist_gap2[min2] < 70):
        board_num.append(min2)
    else:
        board_num.append(10)
    if(hist_gap3[sec1] < 70):
        board_num.append(sec1)
    else:
        board_num.append(10)
    if(hist_gap4[sec2] < 70):
        board_num.append(sec2)
    else:
        board_num.append(10)
    hist_gap1 = []
    hist_gap2 = []
    hist_gap3 = []
    hist_gap4 = []
    for i in range(0,4):
        if(board_num[i] == 10):
            state = False
    time = (600*board_num[0])+(60*board_num[1])+(10*board_num[2])+(board_num[3])
    #print(board_num)
    board_num = []
    return state, time


file_path = input("파일 이름을 입력해주세요: ")
start = time.time()
cap = cv2.VideoCapture(file_path)
if not cap.isOpened():
    print("Could not Open :", file_path)
    exit(0)
print("ok")
fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame/fps
t = 1000
iter_count = 0
time_count = 0
first_time = []
while(time_count < 5):
    target = t*fps
    state,board_time = num_detection(target)
    #print(state, board_time)
    if(state == True):
        time_count += 1
        first_time.append(board_time)
    t += 5
    iter_count += 1
#print(first_time)
first_start = (t - int(iter_count*5/2)) - stats.trim_mean(first_time,0.25)-2.5
#print("Firsthalf_start: ", str(datetime.timedelta(seconds=first_start)))
print("Firsthalf_start: ",first_start)
#print(state, board_time)
t = first_start + 45*60
fail_count = 0
while(fail_count < 4):
    target = t*fps
    state,board_time = num_detection(target)
    if(state == False):
        fail_count += 1
    t += 5
first_end = t - 17.5
#print("Firsthalf_end: ",str(datetime.timedelta(seconds=first_end)))
print("Firsthalf_end: ", first_end)
t = first_end + 10*60
iter_count = 0
time_count = 0
second_time = []
while(time_count < 5):
    target = t*fps
    state,board_time = num_detection(target)
    if(state == True):
        time_count += 1
        second_time.append(board_time)
    t += 5
    iter_count += 1
#print(second_time)
second_start = (t - int(iter_count*5/2)) - (stats.trim_mean(second_time,0.2)- 45*60)
#print("secondhalf_start: ",str(datetime.timedelta(seconds=second_start)))
print("secondhalf_start: ",second_start)
#print(state, board_time)
t = second_start + 45*60
fail_count = 0
while(fail_count < 4):
    target = t*fps
    state,board_time = num_detection(target)
    if(state == False):
        fail_count += 1
    t += 5
second_end = t - 17.5
#print("secondhalf_end: ",str(datetime.timedelta(seconds=second_end)))
print("secondhalf_end: ",second_end)
print("time :", time.time() - start)
cap.release()
exit()