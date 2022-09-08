import numpy as np
import cv2
import os
import time

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
    if bgrResult_num > 70:
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


def main(input):
    file_name = input
    cap = cv2.VideoCapture(file_name)  # 파일 읽어들이기
    fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수 구하기(= 초당 frame 갯수)
    # 전반 시작 찾기
    check_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        temp = check_frame
        cnt1 = 0
        for i in range(0, 10):
            temp += fps * 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, temp)
            ret, frame = cap.read()
            if is_ingame(frame):
                cnt1 += 1
        # cnt2 = 0
        # temp += fps * 60 * 1
        # for i in range(0, 10):
        #     temp += fps * 10
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, temp)
        #     ret, frame = cap.read()
        #     if is_ingame(frame):
        #         cnt2 += 1
        # if cnt1 > 3 and cnt2 > 5:
        if cnt1 > 6 :
            cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame - 10 * fps)
            break
        else:
            check_frame += 10 * fps

    first_half_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(first_half_start_frame // fps)

    # 전반 종료 찾기
    check_frame = first_half_start_frame
    check_frame += fps * 60 * 45
    while True:
        cnt = 0
        check_frame += 20 * fps
        for i in range(-5, 5):
            temp = check_frame + 3 * i * fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, temp)
            ret, frame = cap.read()
            if is_ingame(frame):
                cnt += 1
        if cnt < 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
            break

    first_half_end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    print(first_half_end_frame // fps)
    # 후반 시작 찾기
    start_frame = first_half_end_frame + fps * 60 * 5
    check_frame = first_half_end_frame
    while True:
        temp = check_frame
        cnt1 = 0
        for i in range(0, 5):
            temp += fps * 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, temp)
            ret, frame = cap.read()
            if is_ingame(frame):
                cnt1 += 1
        cnt2 = 0
        temp += fps * 60 * 5
        for i in range(0, 5):
            temp += fps * 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, temp)
            ret, frame = cap.read()
            if is_ingame(frame):
                cnt2 += 1
        if cnt1 > 3 and cnt2 > 3:
            cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame - 10 * fps)
            break
        else:
            check_frame += 10 * fps

    second_half_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(second_half_start_frame // fps)

    # 후반 종료 찾기
    check_frame = second_half_start_frame
    check_frame += fps * 60 * 45
    while True:
        cnt = 0
        check_frame += 20 * fps
        for i in range(-5, 5):
            temp = check_frame + 3 * i * fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, temp)
            ret, frame = cap.read()
            if is_ingame(frame):
                cnt += 1
        if cnt < 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
            break

    second_half_end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(second_half_end_frame // fps)

    first_half_start_time = first_half_start_frame / fps
    first_half_end_time = first_half_end_frame / fps
    second_half_start_time = second_half_start_frame / fps
    second_half_end_time = second_half_end_frame / fps


    cap.release()
    cv2.destroyAllWindows()
    return round(first_half_start_time), round(first_half_end_time), round(second_half_start_time), round(second_half_end_time)


# start = time.time()
# input  = "tving_video_224/P470472958_EPI0118_01_t35.mp4"
# a, b, c, d = main(input)
# end = time.time()
# print(end - start)
# result_1 = "ffmpeg -i " + input + " -ss " + str(a) + " -t " + str(b - a) + " -vcodec copy -acodec copy before_half.mp4"
# os.system(result_1)
# result_2 = "ffmpeg -i " + input + " -ss " + str(c) + " -t " + str(d - c) + " -vcodec copy -acodec copy after_half.mp4"
# os.system(result_2)
