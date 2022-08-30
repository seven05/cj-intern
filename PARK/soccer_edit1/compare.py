import time
import cv2
import numpy as np
import csv
'''
x = 1000
img = cv2.imread('Crop2\%d_1.jpg'%x,0)
rows,cols = img.shape
print(rows,cols)

number1_array = []
for k in range(0,9):
    count = 0
    temp_img = cv2.imread('number2\min1\%d.jpg'%k,0)
    for i in range(rows):
        for j in range(cols):
            #print(temp_img[i,j])
            if(abs(img[i,j]-temp_img[i,j]) > 128):
                count +=1
    number1_array.append(count)
print(number1_array)
number1 = np.argmin(number1_array)
print(number1)
'''
'''
x_array = []
x_hist1 = []
np.zeros(x_hist1)
for k in range(0,9):
    for i in range(rows):
        for j in range(cols):
            if(img[i,j] > 128):
                x_hist1[i] += 1

for k in range(0,9):
    temp_img = cv2.imread('number2\min1\%d.jpg'%k,0)
x_hist2 = []
np.zeros(x_hist2)
'''
def histogram(filepath):
    #print(filepath)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
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

hist_min1_num0 = histogram("number\min1/0.jpg")
hist_min1_num1 = histogram("number\min1/1.jpg")
hist_min1_num2 = histogram("number\min1/2.jpg")
hist_min1_num3 = histogram("number\min1/3.jpg")
hist_min1_num4 = histogram("number\min1/4.jpg")
hist_min1_num5 = histogram("number\min1/5.jpg")
hist_min1_num6 = histogram("number\min1/6.jpg")
hist_min1_num7 = histogram("number\min1/7.jpg")
hist_min1_num8 = histogram("number\min1/8.jpg")
hist_min1_num9 = histogram("number\min1/9.jpg")
hist_min2_num0 = histogram("number\min2/0.jpg")
hist_min2_num1 = histogram("number\min2/1.jpg")
hist_min2_num2 = histogram("number\min2/2.jpg")
hist_min2_num3 = histogram("number\min2/3.jpg")
hist_min2_num4 = histogram("number\min2/4.jpg")
hist_min2_num5 = histogram("number\min2/5.jpg")
hist_min2_num6 = histogram("number\min2/6.jpg")
hist_min2_num7 = histogram("number\min2/7.jpg")
hist_min2_num8 = histogram("number\min2/8.jpg")
hist_min2_num9 = histogram("number\min2/9.jpg")
hist_sec1_num0 = histogram("number\sec1/0.jpg")
hist_sec1_num1 = histogram("number\sec1/1.jpg")
hist_sec1_num2 = histogram("number\sec1/2.jpg")
hist_sec1_num3 = histogram("number\sec1/3.jpg")
hist_sec1_num4 = histogram("number\sec1/4.jpg")
hist_sec1_num5 = histogram("number\sec1/5.jpg")
hist_sec2_num0 = histogram("number\sec2/0.jpg")
hist_sec2_num1 = histogram("number\sec2/1.jpg")
hist_sec2_num2 = histogram("number\sec2/2.jpg")
hist_sec2_num3 = histogram("number\sec2/3.jpg")
hist_sec2_num4 = histogram("number\sec2/4.jpg")
hist_sec2_num5 = histogram("number\sec2/5.jpg")
hist_sec2_num6 = histogram("number\sec2/6.jpg")
hist_sec2_num7 = histogram("number\sec2/7.jpg")
hist_sec2_num8 = histogram("number\sec2/8.jpg")
hist_sec2_num9 = histogram("number\sec2/9.jpg")

start = time.time()
test = []

for x in range(1000,1020):
    tmp = []
    min1_hist = histogram('Crop2\%d_1.jpg'%x)
    min2_hist = histogram('Crop2\%d_2.jpg'%x)
    sec1_hist = histogram('Crop2\%d_3.jpg'%x)
    sec2_hist = histogram('Crop2\%d_4.jpg'%x)
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
    print(hist_gap1)
    # print(hist_gap2)
    # print(hist_gap3)
    # print(hist_gap4)
    if(hist_gap1[min1] < 70):
        tmp.append(min1)
    else:
        tmp.append(10)
    if(hist_gap2[min2] < 70):
        tmp.append(min2)
    else:
        tmp.append(10)
    if(hist_gap3[sec1] < 70):
        tmp.append(sec1)
    else:
        tmp.append(10)
    if(hist_gap4[sec2] < 70):
        tmp.append(sec2)
    else:
        tmp.append(10)
    test.append(tmp)
    hist_gap1 = []
    hist_gap2 = []
    hist_gap3 = []
    hist_gap4 = []
    tmp = []
print("time :", time.time() - start)
print(test)