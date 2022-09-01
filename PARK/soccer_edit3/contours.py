import os
from platform import win32_edition
import cv2
import time
import numpy as np

MIN_AREA, MAX_AREA = 200,500
MIN_WIDTH, MIN_HEIGHT = 0.5, 3
MIN_RATIO, MAX_RATIO = 0.6, 0.9
MAX_DIAG_MULTIPLYER = 3
MAX_ANGLE_DIFF = 1.0
MAX_AREA_DIFF = 0.2
MAX_WIDTH_DIFF = 0.15
MAX_HEIGHT_DIFF = 0.15
MIN_N_MATCHED = 4

try:
    if not os.path.exists('./number'):
        os.makedirs('./number')
except OSError:
    print('error : creating dir number')

file_path = input("파일 이름을 입력해주세요: ")
cap = cv2.VideoCapture(file_path)
if not cap.isOpened():
    print("Could not Open :", file_path)
    exit(0)
print("file ok")

fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame/fps

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

for t in range(1500,6400):
    target = t*fps
    cap.set(cv2.CAP_PROP_POS_FRAMES,target)
    ret, img = cap.read()
    height, width, channel = img.shape
    # print(height,width,channel)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_crop = img[0:int(height/4),0:int(width/4)]
    height, width, channel = img_crop.shape
    # cv2.imshow("img",img_crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    '''
    img_gray_thr = cv2.adaptiveThreshold(
        img_gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    # cv2.imshow("img",img_gray_thr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    '''
    img_blur = cv2.GaussianBlur(img_gray, ksize=(9,9), sigmaX=0)
    img_blur_thr = cv2.adaptiveThreshold(
        img_blur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    # cv2.imshow("img",img_blur_thr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(
        img_blur_thr,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    img_contours = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(img_contours, contours=contours, contourIdx=-1, color=(255,255,255))
    # cv2.imshow("img",img_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    # cv2.imshow("img",img_boundingbox)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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

    # cv2.imshow("img",img_possiblebox)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
        
    img_possiblebox_filter = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            cv2.rectangle(img_possiblebox_filter, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)

    # cv2.imshow("img",img_possiblebox_filter)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    number_list = []
    for r in matched_result:
        for d in r:
            temp = []
            temp.append(d['x'])
            temp.append(d['y'])
            temp.append(d['w'])
            temp.append(d['h'])
            number_list.append(temp)
    # print(number_list)
    # print(len(number_list))
    # print(matched_result)

    if(len(number_list) != 4):
        continue
    
    number_list.sort(key=lambda x:x[0])
    #print(number_list)
    
    for i in range(len(number_list)):
        x = number_list[i][0]
        y = number_list[i][1]
        w = number_list[i][2]
        h = number_list[i][3]
        cv2.imwrite('./number' + "/%d_num_%d.jpg" %(t,i), img_crop[y:y+h,x:x+w])