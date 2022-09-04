import time
import cv2
import numpy as np

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

hist_num0 = histogram("number/0.jpg")
hist_num1 = histogram("number/1.jpg")
hist_num2 = histogram("number/2.jpg")
hist_num3 = histogram("number/3.jpg")
hist_num4 = histogram("number/4.jpg")
hist_num5 = histogram("number/5.jpg")
hist_num6 = histogram("number/6.jpg")
hist_num7 = histogram("number/7.jpg")
hist_num8 = histogram("number/8.jpg")
hist_num9 = histogram("number/9.jpg")

#start = time.time()
test = []

for x in range(1500,1501):
    tmp = []
    min1_hist = histogram('Crop2\%d_0.jpg'%x)
    min2_hist = histogram('Crop2\%d_1.jpg'%x)
    sec1_hist = histogram('Crop2\%d_2.jpg'%x)
    sec2_hist = histogram('Crop2\%d_3.jpg'%x)
    hist_gap1 = []
    for i in range(0,10):
        temp1 = [abs(globals()[f'hist_num{i}'][j] - min1_hist[j]) for j in range(len(min1_hist))]
        temp2 = [abs(globals()[f'hist_num{i}'][j] - min2_hist[j]) for j in range(len(min2_hist))]
        temp4 = [abs(globals()[f'hist_num{i}'][j] - sec2_hist[j]) for j in range(len(sec1_hist))]
        hist_gap1.append(sum(temp1))
        hist_gap2.append(sum(temp2))
        hist_gap4.append(sum(temp4))
        temp1 = []
        temp2 = []
        temp4 = []
    for i in range (0,6):
        temp3 = [abs(globals()[f'hist_num{i}'][j] - sec1_hist[j]) for j in range(len(sec1_hist))]
        hist_gap3.append(sum(temp3))
        temp3 = []
    min1 = np.argmin(hist_gap1)
    min2 = np.argmin(hist_gap2)
    sec1 = np.argmin(hist_gap3)
    sec2 = np.argmin(hist_gap4)
    print(hist_gap1)
    print(hist_gap2)
    print(hist_gap3)
    print(hist_gap4)
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
#print("time :", time.time() - start)
print(test)