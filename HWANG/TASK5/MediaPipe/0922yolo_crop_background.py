import time
start = time.time()
import cv2
import my_detection
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def video_selfi_segmentation(input_dir, input_name, output_name):
    file_directory = input_dir
    file_name = input_name

    outfile_name = output_name + '.avi'
    file_name = file_directory + file_name +'.mp4'

    bg_image = cv2.imread('./LJM/background_2.png')
    bg_image = cv2.resize(bg_image, (640, 480), cv2.INTER_LANCZOS4)
    h, w, _ =bg_image.shape
    cap = cv2.VideoCapture(file_name)

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D', 'I', 'V', 'X'

    out = cv2.VideoWriter(outfile_name, fourcc, fps, (w, h))

    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:

      while cap.isOpened():
        success, image = cap.read()
        if not success:
          break

        if my_detection.my_detect(image) is None:
            continue

        # Convert the BGR image to RGB.
        temp_image = np.zeros(bg_image.shape, dtype=np.uint8)
        temp_image[:] = bg_image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        temp_image[120:360, 250:400] = image

        image.flags.writeable = False
        results = selfie_segmentation.process(temp_image)

        image.flags.writeable = True
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR)

        condition = np.stack(
          (results.segmentation_mask,) * 3, axis=-1) > 0.8

        output_image = np.where(condition, temp_image, bg_image)

        out.write(output_image)
        cv2.imshow('MediaPipe Selfie Segmentation', output_image)
        if cv2.waitKey(5) & 0xFF == 27:
          break

    cap.release()
    out.release()
    end = time.time()
    return end - start

video_selfi_segmentation('./LJM/', 'ufc_crop_yolo', './yolo_crop_background')