import cv2
import my_detection
import mediapipe as mp
import numpy as np
import time
def crop(input_dir, input_name, output_name):
    start = time.time()
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
        x_min, y_min, height, width = my_detection.my_detect(image)
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

crop('./LJM/', 'cv_camera_sensor_stream_handler', './only_face')