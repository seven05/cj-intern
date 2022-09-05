import cv2
import os
import time
import numpy as np

file_path = input("파일 이름을 입력해주세요: ")
start = time.time()
cap = cv2.VideoCapture("Video/"+file_path)
if not cap.isOpened():
    print("Could not Open :", file_path)
    exit(0)
print("file ok")

fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame/fps

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

MIN_AREA, MAX_AREA = 200,500
MIN_WIDTH, MIN_HEIGHT = 0.5, 3
MIN_RATIO, MAX_RATIO = 0.6, 0.9
MAX_DIAG_MULTIPLYER = 3
MAX_ANGLE_DIFF = 1.0
MAX_AREA_DIFF = 0.2
MAX_WIDTH_DIFF = 0.15
MAX_HEIGHT_DIFF = 0.15
MIN_N_MATCHED = 4

def find_chars(contour_list):
        matched_result_idx = []
        
        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue
                    
                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])
                
                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
                
                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']
                
                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])
                    
            matched_contours_idx.append(d1['idx'])
            
            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue
                
            matched_result_idx.append(matched_contours_idx)
            
            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])
            
            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
            
            recursive_contour_list = find_chars(unmatched_contour)
            
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)
                
            break
            
        return matched_result_idx

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

def num_detection(target):
    state = True
    board_num = []
    cap.set(cv2.CAP_PROP_POS_FRAMES,target)
    ret, img = cap.read()
    height, width, channel = img.shape
    img_crop = img[0:int(height/4),0:int(width/4)]
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, ksize=(9,9), sigmaX=0)
    img_blur_thr = cv2.adaptiveThreshold(
        img_blur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    contours, _ = cv2.findContours(
        img_blur_thr,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    img_contours = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(img_contours, contours=contours, contourIdx=-1, color=(255,255,255))
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
    possible_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        
        if MIN_AREA < area < MAX_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
    img_possiblebox = np.zeros((height, width, channel), dtype = np.uint8)
    for d in possible_contours:
        cv2.rectangle(img_possiblebox, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
        
    img_possiblebox_filter = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            cv2.rectangle(img_possiblebox_filter, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)
    number_list = []
    for r in matched_result:
        for d in r:
            temp = []
            temp.append(d['x'])
            temp.append(d['y'])
            temp.append(d['w'])
            temp.append(d['h'])
            number_list.append(temp)
    if(len(number_list) != 4):
        state = False
        return state, 0
    number_list.sort(key=lambda x:x[0])
    for i in range(len(number_list)):
        x = number_list[i][0]
        y = number_list[i][1]
        w = number_list[i][2]
        h = number_list[i][3]
        globals()[f'crop_num{i}'] = img_crop[y:y+h,x:x+w]
    for i in range(0,4):
        h, w = globals()[f'crop_num{i}'].shape
        if(globals()[f'crop_num{i}'][0][0] > 128 and globals()[f'crop_num{i}'][h-1][0] > 128 and globals()[f'crop_num{i}'][0][w-1] > 128 and globals()[f'crop_num{i}'][h-1][w-1] > 128):
            globals()[f'thr_num{i}'] = cv2.threshold(globals()[f'crop_num{i}'], 128, 0, cv2.THRESH_BINARY)[1]
        else:
            globals()[f'thr_num{i}'] = cv2.threshold(globals()[f'crop_num{i}'], 128, 255, cv2.THRESH_BINARY)[1]
        x_min,y_min,x_max,y_max = edge_cut(f'thr_num{i}')
        if(x_min == x_max or y_min == y_max):
            state = False
            return state, 0
        globals()[f'edge_cut_num{i}'] = globals()[f'thr_num{i}'][y_min:y_max,x_min:x_max]
        globals()[f'resize_num{i}'] = cv2.resize(globals()[f'edge_cut_num{i}'],dsize=(48,48))
    min1_hist = histogram(globals()['resize_num0'])
    min2_hist = histogram(globals()['resize_num1'])
    sec1_hist = histogram(globals()['resize_num2'])
    sec2_hist = histogram(globals()['resize_num3'])
    hist_gap1 = []
    hist_gap2 = []
    hist_gap3 = []
    hist_gap4 = []
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
    if(hist_gap1[min1] < 600):
        board_num.append(min1)
    else:
        board_num.append(10)
    if(hist_gap2[min2] < 600):
        board_num.append(min2)
    else:
        board_num.append(10)
    if(hist_gap3[sec1] < 600):
        board_num.append(sec1)
    else:
        board_num.append(10)
    if(hist_gap4[sec2] < 600):
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

t = 1000
iter_count = 0
time_count = 0
first_time = []
while(time_count < 5):
    target = t*fps
    state,board_time = num_detection(target)
    if(state == True):
        time_count += 1
        first_time.append(board_time)
    t += 5
    iter_count += 1
first_start = (t - int(iter_count*5/2)) - stats.trim_mean(first_time,0.25)-2.5
print("Firsthalf_start: ",first_start)
t = first_start + 45*60
fail_count = 0
while(fail_count < 4):
    target = t*fps
    state,board_time = num_detection(target)
    if(state == False):
        fail_count += 1
    t += 5
first_end = t - 17.5
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
second_start = (t - int(iter_count*5/2)) - (stats.trim_mean(second_time,0.2)- 45*60)
print("secondhalf_start: ",second_start)
t = second_start + 45*60
fail_count = 0
while(fail_count < 4):
    target = t*fps
    state,board_time = num_detection(target)
    if(state == False):
        fail_count += 1
    t += 5
second_end = t - 17.5
print("secondhalf_end: ",second_end)
print("time :", time.time() - start)
cap.release()
exit()