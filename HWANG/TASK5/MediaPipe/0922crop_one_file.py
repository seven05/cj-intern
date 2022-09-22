import cv2
import mediapipe as mp
import math
import time
import numpy as np
start = time.time()
# input is an image, and return the x_min, x_max, height, width
# 지금은 detection된 것들 중 하나를 추출하지만, 필요에 따라 classification을 사용하여 원하는 objcect에 대해서만
# return 할 수 있도록 수정할 수 있도록 할 예정.
# 0922 추가 : 일단 bounding box가 가장 큰 object의 좌표를 가져오기로 함.

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def my_detect(image):
    x_min, y_min = 0, 0
    width, height = 0, 0
    H, W, _ = image.shape
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # # Draw face detections of each face.
        if not results.detections:
            return -1, -1, -1, -1

        size = -1
        for detection in results.detections:
            location_data = detection.location_data
            bb = location_data.relative_bounding_box
            if size < bb.width * bb.height:
                x_min, y_min, width, height = bb.xmin, bb.ymin, bb.width, bb.height
                size = bb.width * bb.height

    return max(0, math.floor(x_min * W)), max(0, math.floor(y_min * H)), math.ceil(height * H), math.ceil(width * W)

def crop(input_dir, input_name, output_name):
    file_directory = input_dir
    file_name = input_name

    outfile_name = output_name + '.avi'
    file_name = file_directory + file_name +'.mp4'

    cap = cv2.VideoCapture(file_name)
    result_image = np.zeros((350, 350, 3), dtype=np.uint8)
    h, w, _ = result_image.shape  # 고정됨
    result_image[:] = (256, 256, 256)

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D', 'I', 'V', 'X'

    out = cv2.VideoWriter(outfile_name, fourcc, fps, (w, h))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        x_min, y_min, height, width = my_detect(image)
        if x_min < 0 or y_min < 0 or height <= 0 or width <= 0:
            out.write(result_image)
            continue

        x_mid = h // 2 # 150
        y_mid = w // 2 # 150
        diff_x = x_mid - (width // 2)
        diff_y = y_mid - (height // 2)
        start_x = x_min - diff_x
        start_y = y_min - diff_y
        result_image = image[start_y:start_y+h, start_x:start_x+w]
        # result_image[:] = image[0:350, 0:350]
        # if result_image.shape != (h, w, 3):
        #     result_image = cv2.resize(result_image, (w, h), cv2.INTER_NEAREST)
        cv2.imshow('CROP using MediaPipe face detection', result_image)
        # print(result_image.shape)
        out.write(result_image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    end = time.time()
    print("total running time is", end-start)
    cap.release()
    out.release()
    return end-start

crop('./LJM/', 'corry_b2', 'only_face')