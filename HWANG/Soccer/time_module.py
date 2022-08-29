import numpy as np
import cv2
import os
import time


def main(input):
    file_name = input
    cap = cv2.VideoCapture(file_name)  # 파일 읽어들이기

    fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수 구하기(= 초당 frame 갯수)

    now = cap.get(cv2.CAP_PROP_POS_FRAMES) # 현재 프레임
    first_half_start_frame = find_start(now)
    first_half_end_frame = find_end(first_half_start_frame)
    start_frame = first_half_end_frame + fps * 60 * 2
    second_half_start_frame = find_start(start_frame)
    second_half_end_frame = find_end(second_half_start_frame)


    first_half_start_time = first_half_start_frame / fps
    first_half_end_time = first_half_end_frame / fps
    second_half_start_time = second_half_start_frame / fps
    second_half_end_time = second_half_end_frame / fps


    cap.release()
    cv2.destroyAllWindows()
    return first_half_start_time, first_half_end_time, second_half_start_time, second_half_end_time


# img를 받아 RGB를 이용한 색 추출(하얀색 배경을 제외하면 검은색으로 표시)
# 출처 : https://engineer-mole.tistory.com/236
# 이후 이중 for 문을 돌며 white가 얼마나 많은지 확인(로고와 비슷한지 확인하기 위해)
# white의 갯수가 일정 수치 이상이라면 True, 아니라면 False
def is_ingame(image):
    # print(image.shape)
    height = round(image.shape[0] / 6)
    width = round(image.shape[1] / 6)
    image_crop = image[0:height, 0:width].copy()

    # BGR로 색추출
    bgrLower = np.array([200, 200, 200])  # 추출할 색의 하한
    bgrUpper = np.array([225, 225, 225])  # 추출할 색의 상한
    bgrResult = bgrExtraction(image_crop, bgrLower, bgrUpper)

    # 총 하얀점 갯수가 몇 개인지 계산
    bgrResult_num = whilte_num(bgrResult)
    # print(bgrResult_num)
    if bgrResult_num > 80: # 이 부분도 화질이 달라y질 것을 고려하면 비율로 하는것이 좋을 듯
        return True
    else:
        return False

# BGR로 특정 색을 추출하는 함수
def bgrExtraction(image, bgrLower, bgrUpper):
    img_mask = cv2.inRange(image, bgrLower, bgrUpper)
    result = cv2.bitwise_and(image, image, mask=img_mask)

    return result

# 이중 for 문을 돌며 white가 얼마나 많은지 확인(로고와 비슷한지 확인하기 위해)
def whilte_num(image):
    white = 199 # 3차원 모두 비교할거면 np.ndarray([x, y, z])
    height, width, _ = image.shape
    num = 0
    for y in range(0, height):
        for x in range(0, width):
            pt = (y, x)
            if image[pt][0] > white:
                num += 1
    return num

# 시작 지점 찾기
# 1분마다 6초에 대해 초당 한 프레임 씩 내가 원하는 조건이 되는지 확인
# 6개의 프레임 중 6개 모두 만족하면 시작이라고 인식
def find_start(check_frame):
    while(True):
        cnt = 0
        check_frame += 5 * fps
        for i in range(0, 3):
            temp = check_frame + i * fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, temp)
            ret, frame = cap.read()
            if is_ingame(frame):
                cnt += 1
        if cnt == 3:
            cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - (20 * fps))
#    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES))
    return cap.get(cv2.CAP_PROP_POS_FRAMES)


# 끝나는 지점 찾기
# 인풋 프레임으로부터 45분에 대한 프레임을 더한 후
# 매 1분마다 20초에 대해 초당 한 프헤임씩 내가 원하는 조건이 되는지 확인
# 20프레임 중 모두 조건에 만족하면 경기 종료라고 인식
def find_end(check_frame):
    check_frame += fps * 60 * 45
    while(True):
        cnt = 0
        check_frame += 10 * fps
        for i in range(0, 10):
            temp = check_frame + i * fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, temp)
            ret, frame = cap.read()
            if is_ingame(frame):
                cnt += 1
        if cnt == 0:
            break

    # ret, frame = cap.read()
    # cv2.imshow("result", frame)
    # cv2.waitKey()
    return cap.get(cv2.CAP_PROP_POS_FRAMES)