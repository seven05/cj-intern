import time
start = time.time()
import cv2
import my_detection
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def blur(input_dir, input_name, output_name):
    file_directory = input_dir
    file_name = input_name

    outfile_name = output_name + '.avi'
    file_name = file_directory + file_name +'.mp4'

    cap = cv2.VideoCapture(file_name)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D', 'I', 'V', 'X'

    out = cv2.VideoWriter(outfile_name, fourcc, fps, (w, h))

    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          break

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bg_image = cv2.GaussianBlur(image, (55, 55), 0)
        condition = np.stack(
          (results.segmentation_mask,) * 3, axis=-1) > 0.8

        output_image = np.where(condition, image, bg_image)

        out.write(output_image)
        cv2.imshow('MediaPipe Selfie Segmentation', output_image)
        if cv2.waitKey(5) & 0xFF == 27:
          break

    cap.release()
    out.release()
    end = time.time()
    return end - start

blur('./LJM/', 'cv_camera_sensor_stream_handler', './blur')