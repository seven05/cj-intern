from sre_constants import SUCCESS
import cv2
import os
import time
import numpy as np
from scipy import stats

def histogram(img):                             #숫자인식에 쓰일 히스토그램 함수
    #print(filepath)                            #x,y축으로 숫자 픽셀 갯수 기록
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

#전광판에서 시간을 나타내는 숫자를 검출하기위한 상수값들
MIN_AREA, MAX_AREA = 200,500
MIN_WIDTH, MIN_HEIGHT = 0.5, 3
MIN_RATIO, MAX_RATIO = 0.6, 0.9
MAX_DIAG_MULTIPLYER = 3
MAX_ANGLE_DIFF = 1.0
MAX_AREA_DIFF = 0.2
MAX_WIDTH_DIFF = 0.15
MAX_HEIGHT_DIFF = 0.15
MIN_N_MATCHED = 4
# 숫자 0~9까지 표본
hist_num0 = histogram(cv2.imread("number/0.jpg",cv2.IMREAD_GRAYSCALE))
hist_num1 = histogram(cv2.imread("number/1.jpg",cv2.IMREAD_GRAYSCALE))
hist_num2 = histogram(cv2.imread("number/2.jpg",cv2.IMREAD_GRAYSCALE))
hist_num3 = histogram(cv2.imread("number/3.jpg",cv2.IMREAD_GRAYSCALE))
hist_num4 = histogram(cv2.imread("number/4.jpg",cv2.IMREAD_GRAYSCALE))
hist_num5 = histogram(cv2.imread("number/5.jpg",cv2.IMREAD_GRAYSCALE))
hist_num6 = histogram(cv2.imread("number/6.jpg",cv2.IMREAD_GRAYSCALE))
hist_num7 = histogram(cv2.imread("number/7.jpg",cv2.IMREAD_GRAYSCALE))
hist_num8 = histogram(cv2.imread("number/8.jpg",cv2.IMREAD_GRAYSCALE))
hist_num9 = histogram(cv2.imread("number/9.jpg",cv2.IMREAD_GRAYSCALE))

#cv2함수인 contour로 찾아낸 경계선중에 내가 원하는 숫자모양만 걸러내기 
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
            # 숫자끼리는 비슷해야하므로 각도나 너비 높이가 비슷해야함
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
                
        matched_contours_idx.append(d1['idx'])
        #숫자 4개가 뭉쳐있어야함
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

#resize하기전에 숫자영역의 여백이 히스토그램 오차를 내는것을 막기위해 여백을 자름
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

#원하는 시간을 넣으면 숫자인식을 진행하는 함수
def num_detection(target):
    state = True        # 숫자가 정상인지 상태표시
    board_num = []
    cap.set(cv2.CAP_PROP_POS_FRAMES,target)     
    ret, img = cap.read()
    height, width, channel = img.shape
    img_crop = img[0:int(height/4),0:int(width/4)]      # 빠른처리를 위해 왼쪽 상단 1/16만 사용
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)       
    img_blur = cv2.GaussianBlur(img_gray, ksize=(9,9), sigmaX=0)        #노이즈 제거를 위한 가우시안필터
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
    global possible_contours
    possible_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        # 최소 최대 영역을 제한하는 과정인데 하드코딩에 가까워서 개선하고싶음
        # 가로세로 비율만으로는 비슷한 오브젝트가 너무 많았음
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
    for r in matched_result:        #숫자라고 판단된 오브젝트의 x,y,w,h값만을 따로 담는 과정
        for d in r:
            temp = []
            temp.append(d['x'])
            temp.append(d['y'])
            temp.append(d['w'])
            temp.append(d['h'])
            number_list.append(temp)
    if(len(number_list) < 4):       #숫자라고 판단되는 오브젝트가 4개 미만이면 잘못된 결과
        state = False
        return state, 0
    number_list.sort(key=lambda x:x[0])
    for i in range(4):              #숫자오브젝트가 왼쪽에 가까울 확률이 높으니 왼쪽으로부터 4개의 오브젝트를 사용
        x = number_list[i][0]
        y = number_list[i][1]
        w = number_list[i][2]
        h = number_list[i][3]
        globals()[f'crop_num{i}'] = img_crop[y:y+h,x:x+w]
    for i in range(0,4):
        img_crop_num = globals()[f'crop_num{i}']
        img_crop_num_gray = cv2.cvtColor(img_crop_num, cv2.COLOR_BGR2GRAY)
        h,w = img_crop_num_gray.shape
        #4개의 꼭지점에 있는 픽셀값이 모두 하얀색으로 나온다면 추가시간의 전광판 숫자라고 판단해서 색반전
        if(img_crop_num_gray[0][0] > 128 and img_crop_num_gray[h-1][0] > 128 and img_crop_num_gray[0][w-1] > 128 and img_crop_num_gray[h-1][w-1] > 128):
            globals()[f'thr_num{i}'] = cv2.threshold(img_crop_num_gray, 128, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            globals()[f'thr_num{i}'] = cv2.threshold(img_crop_num_gray, 128, 255, cv2.THRESH_BINARY)[1]
        #얻어낸 숫자의 여백을 제거
        x_min,y_min,x_max,y_max = edge_cut(globals()[f'thr_num{i}'])
        if(x_min == x_max or y_min == y_max):
            globals()[f'edge_cut_num{i}'] = globals()[f'thr_num{i}']
        else:
            globals()[f'edge_cut_num{i}'] = globals()[f'thr_num{i}'][y_min:y_max,x_min:x_max]
        #여백을 제거한 숫자이미지를 히스토그램 비교를 위해 같은 크기로 만듬
        globals()[f'resize_num{i}'] = cv2.resize(globals()[f'edge_cut_num{i}'],dsize=(48,48))
    min1_hist = histogram(globals()['resize_num0'])
    min2_hist = histogram(globals()['resize_num1'])
    sec1_hist = histogram(globals()['resize_num2'])
    sec2_hist = histogram(globals()['resize_num3'])
    hist_gap1 = []
    hist_gap2 = []
    hist_gap3 = []
    hist_gap4 = []
    
    for i in range(0,10):   #10의자리 분, 1의자리 분, 1의자리 초 는 0~9가 모두 나오므로 10번
        temp1 = [abs(globals()[f'hist_num{i}'][j] - min1_hist[j]) for j in range(len(min1_hist))]
        temp2 = [abs(globals()[f'hist_num{i}'][j] - min2_hist[j]) for j in range(len(min2_hist))]
        temp4 = [abs(globals()[f'hist_num{i}'][j] - sec2_hist[j]) for j in range(len(sec1_hist))]
        hist_gap1.append(sum(temp1))
        hist_gap2.append(sum(temp2))
        hist_gap4.append(sum(temp4))
        temp1 = []
        temp2 = []
        temp4 = []
    for i in range (0,6):   #10의자리 초는 0~5만 나오므로 6번 
        temp3 = [abs(globals()[f'hist_num{i}'][j] - sec1_hist[j]) for j in range(len(sec1_hist))]
        hist_gap3.append(sum(temp3))
        temp3 = []
    min1 = np.argmin(hist_gap1)    # 10개의 표본숫자와 가장 작은 오차를 보이는 숫자로 판단
    min2 = np.argmin(hist_gap2)
    sec1 = np.argmin(hist_gap3)
    sec2 = np.argmin(hist_gap4)
    if(hist_gap1[min1] < 800):     # 히스토그램상으로 48*48의 1/4정도인 800의 오차를 넘어가면 다른 글자라고 판단.
        board_num.append(min1)
    else:
        board_num.append(10)
    if(hist_gap2[min2] < 800):
        board_num.append(min2)
    else:
        board_num.append(10)
    if(hist_gap3[sec1] < 800):
        board_num.append(sec1)
    else:
        board_num.append(10)
    if(hist_gap4[sec2] < 800):
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

def main(file_path):
    #start = time.time()
    global cap                              
    cap = cv2.VideoCapture("Video/"+file_path)
    if not cap.isOpened():                            #파일 유효성검사
        print("Could not Open :", file_path)
        exit(0)
    print("file ok")
    fps = cap.get(cv2.CAP_PROP_FPS)     
    fps = round(fps)
    t = 1500    #영상시작 1500초로 점프
    time_count = 0
    first_start_time = []
    while(time_count < 5):  #시작지점을 찾을때는 유효한 전광판 숫자 5개가 들어올때까지 탐색
        target = t*fps
        state,board_time = num_detection(target)
        if(state == True):
            time_count += 1
            first_start_time.append(t-board_time)
        t += 5              # 5초씩 건너뛰면서 탐색
    first_start = stats.trim_mean(first_start_time,0.25) # 숫자인식 오류를 생각해서 절사평균으로 튀는값 제거
    #print("Firsthalf_start: ",first_start)
    t = first_start + 45*60 #시작시간 + 45분뒤로 점프
    fail_count = 0
    #success_count = 0
    first_end_time = []
    while(fail_count < 30):     #추가시간은 언제 끝날지 모르므로 숫자인식이 안되는 지점을 찾기위해 숫자인식 실패가 30회연속이면 끝지점 측정
        target = t*fps
        state,board_time = num_detection(target)
        if(state == False):
            fail_count += 1
        if(state == True):
            # if(2600 < board_time < 3100):
            #     fail_count = 0
            # else:
            #     fail_count +=1
            fail_count = 0
            if(2700 < board_time < 3100):
                first_end_time.append(board_time)
        t += 2
    first_end = t - fail_count*2
    if(len(first_end_time) == 0):
        first_end = first_end
        if((first_end - (first_start+45*60)) > 400):
            first_end = first_start + 48*60
    else:    
        first_end_time_max = max(first_end_time)
        if((first_end - (first_start+45*60)) > 400):
            first_end = first_start + first_end_time_max
        else:
            if(abs(first_end - (first_start + first_end_time_max)) < 30):
                first_end = (first_end + (first_start + first_end_time_max))/2
    # if(first_end - first_start > 3060):
    #     first_end = first_start + 3000
    #print("Firsthalf_end: ", first_end)
    t = first_end + 10*60
    time_count = 0
    second_start_time = []
    while(time_count < 5):
        target = t*fps
        state,board_time = num_detection(target)
        if(state == True):
            time_count += 1
            second_start_time.append(t-(board_time-45*60))
        t += 5
    second_start = stats.trim_mean(second_start_time,0.2)
    #print("secondhalf_start: ",second_start)
    t = second_start + 45*60
    fail_count = 0
    #success_count = 0
    second_end_time = []
    while(fail_count < 30):
        target = t*fps
        state,board_time = num_detection(target)
        if(state == False):
            fail_count += 1
        if(state == True):
            # if(5300 < board_time < 5800):
            #     fail_count = 0
            # else:
            #     fail_count +=1
            fail_count = 0
            if(5400 < board_time < 5800):
                second_end_time.append(board_time)
        t += 2
    second_end = t - fail_count*2   
    if(len(second_end_time) == 0):
        second_end = second_end
        if((second_end - (second_start+45*60)) > 400):
            second_end = second_start + 48*60
    else:
        second_end_time_max = max(second_end_time)
        if((second_end - (second_start+45*60)) > 400):
            second_end = second_start + second_end_time_max
        else:
            if(abs(second_end - (second_start + second_end_time_max - 45*60)) < 30):
                second_end = (second_end + (second_start + second_end_time_max - 45*60))/2
    # if(second_end - second_start > 3060):
    #     second_end = second_start + 3060
    #print("secondhalf_end: ",second_end)
    #print("time :", time.time() - start)
    cap.release()
    return round(first_start), round(first_end),round(second_start),round(second_end)
