import numpy as np
import cv2
import os
import time

start = time.time()
cap = cv2.VideoCapture('P470472958_EPI0001_01_t35.mp4') # 파일 읽어들이기

fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수 구하기(= 초당 frame 갯수)
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 총 프레임 수

# time = cap.get(cv2.CAP_PROP_POS_MSEC) # 프레임 재생 시간
# frame = cap.get(cv2.CAP_PROP_POS_FRAMES) # 현재 프레임

# 이미지 크롭을 비율로 하기 위한 변수
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 224
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 398
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
xmin, xmax = round(w * 23 / 398), round(w * 45 / 398)
ymin, ymax = round(h * 18 / 224), round(h * 35 / 244)

# img를 받아 RGB를 이용한 색 추출(하얀색 배경을 제외하면 검은색으로 표시)
# 출처 : https://engineer-mole.tistory.com/236
# 이후 이중 for 문을 돌며 white가 얼마나 많은지 확인(로고와 비슷한지 확인하기 위해)
# white의 갯수가 일정 수치 이상이라면 True, 아니라면 False
def is_ingame(image):
    # image_crop = image[18:35, 23:45].copy() #들어온 이미지를 크롭, 이후에 비율로 맞출 예정
    # 비율 버전
    image_crop = image[ymin:ymax, xmin:xmax].copy()
    # print("img shape", image_crop.shape) # 현재 들어온 인풋"image_crop"의 크기 (17, 22)

    # BGR로 색추출
    bgrLower = np.array([210, 210, 210])  # 추출할 색의 하한
    bgrUpper = np.array([225, 225, 225])  # 추출할 색의 상한
    bgrResult = bgrExtraction(image_crop, bgrLower, bgrUpper)

    # 총 하얀점 갯수가 몇개인지 계산
    bgrResult_num = whilte_num(bgrResult)
    # 비율 버전
    h,w,c = image_crop.shape
    total_pixel_num = h * w
    ratio = bgrResult_num / total_pixel_num
    if ratio > 0.185: # 70/374는 대충 0.187
    # if bgrResult_num > 70: # 이 부분도 화질이 달라질 것을 고려하면 비율로 하는것이 좋을 듯
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
    white = 209 # 3차원 모두 비교할거면 np.ndarray([x, y, z])
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
        check_frame += 60 * fps
        for i in range(0,6):
            temp = check_frame + i * fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, temp)
            ret, frame = cap.read()
            if is_ingame(frame):
                cnt += 1
        if cnt > 5:
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - (120 * fps))

    # ret, frame = cap.read()
    # cv2.imshow("result", frame)
    # cv2.waitKey()
    return cap.get(cv2.CAP_PROP_POS_FRAMES)


# 끝나는 지점 찾기
# 인풋 프레임으로부터 45분에 대한 프레임을 더한 후
# 매 1분마다 20초에 대해 초당 한 프헤임씩 내가 원하는 조건이 되는지 확인
# 20프레임 중 모두 조건에 만족하면 경기 종료라고 인식
def find_end(check_frame):
    check_frame += fps * 60 * 45
    while(True):
        cnt = 0
        check_frame += 60 * fps
        for i in range(0, 20):
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


now = cap.get(cv2.CAP_PROP_POS_FRAMES) # 현재 프레임
first_half_start_frame = find_start(now)
first_half_end_frame = find_end(first_half_start_frame)
second_half_start_frame = find_start(first_half_end_frame)
second_half_end_frame = find_end(second_half_start_frame)

print(first_half_start_frame)
print(first_half_end_frame)
print(second_half_start_frame)
print(second_half_end_frame)
first_half_start_time = first_half_start_frame / fps
first_half_end_time = first_half_end_frame / fps
second_half_start_time = second_half_start_frame / fps
second_half_end_time = second_half_end_frame / fps
print("-------------------------------------")
print(first_half_start_time)
print(first_half_end_time)
print(second_half_start_time)
print(second_half_end_time)

# os.system("ffmpeg -i P470472958_EPI0001_01_t35.mp4 -ss") # 덜 작성함 이런 식으로 하면 cmd 명령어(ffmpeg) 사용 가능
cap.release()
cv2.destroyAllWindows()
print(time.time() - start)
# 4.5563271045684814 초 걸림
# 5.558298826217651 초 걸림
# 8/25일 10시 기준으로 평균 5초 정도 걸림

# 빨간줄을 로그로 뱉지만 
result_1 = "ffmpeg -i P470472958_EPI0001_01_t35.mp4 -ss " + str(first_half_start_time) + " -t " + str(first_half_end_time) + " -vcodec copy -acodec copy before_half.mp4"
os.system(result_1)
result_2 = "ffmpeg -i P470472958_EPI0001_01_t35.mp4 -ss " + str(second_half_start_time) + " -t " + str(second_half_end_time) + " -vcodec copy -acodec copy after_half.mp4"
os.system(result_2)
# os.system("type temp.txt") # temp.txt를 출력함..
