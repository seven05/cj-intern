import numpy as np
import cv2
import os
import time
# 8-25.py 파일을 한 프레임 당으로 진행 과정을 설명하기 위한 코드

file_name = "frame/6.jpg"

# img를 받아 RGB를 이용한 색 추출(하얀색 배경을 제외하면 검은색으로 표시)
# 출처 : https://engineer-mole.tistory.com/236
# 이후 이중 for 문을 돌며 white가 얼마나 많은지 확인(로고와 비슷한지 확인하기 위해)
# white의 갯수가 일정 수치 이상이라면 True, 아니라면 False
def is_ingame(image):
    image_crop = image[18:35, 23:45].copy() #들어온 이미지를 크롭, 이후에 비율로 맞출 예정

    for_imshow = cv2.resize(image_crop, (398, 224))
    cv2.imshow("before pretreatment", for_imshow)
    cv2.waitKey()

    # BGR로 색추출
    bgrLower = np.array([210, 210, 210])  # 추출할 색의 하한
    bgrUpper = np.array([225, 225, 225])  # 추출할 색의 상한
    bgrResult = bgrExtraction(image_crop, bgrLower, bgrUpper)

    # 총 하얀점 갯수가 몇개인지 계산
    bgrResult_num = whilte_num(bgrResult)
    # 비율 버전
    # h,w,c = image_crop.shape
    # total_pixel_num = h * w
    # ratio = bgrResult_num / total_pixel_num
    # if ratio > 0.185: # 70/374는 대충 0.187
    print("bgrResult_num is : ", bgrResult_num)
    if bgrResult_num > 70: # 이 부분도 화질이 달라질 것을 고려하면 비율로 하는것이 좋을 듯
        return True
    else:
        return False

# BGR로 특정 색을 추출하는 함수
def bgrExtraction(image, bgrLower, bgrUpper):
    img_mask = cv2.inRange(image, bgrLower, bgrUpper)
    result = cv2.bitwise_and(image, image, mask=img_mask)
    for_imshow = cv2.resize(result, (398, 224))
    cv2.imshow("bgrExtraction", for_imshow)
    cv2.waitKey()
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

img = cv2.imread(file_name)
cv2.imshow("img", img)
cv2.waitKey()
is_ingame(img)

cv2.destroyAllWindows()
