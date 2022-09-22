import time
start = time.time()
import cv2
import my_detection
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def video_selfi_segmentation(input_dir, input_name, output_name):
    BG_COLOR = (0, 0, 0)
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
      bg_image = None
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          break

        if my_detection.my_detect(image) is None:
            continue
        x_min, y_min, height, width = my_detection.my_detect(image)
        print(x_min, y_min, width, height)
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_crop = image[y_min:y_min+height, x_min:x_min+width]
        image.flags.writeable = False
        results = selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # bg_image = np.where(, image, BG_COLOR)
        condition = np.stack(
          (results.segmentation_mask,) * 3, axis=-1) > 0.8

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            # bg_image[x_min, x_max, y_min, y_max] = image else BG_COLOR
        output_image = np.where(condition, image, bg_image)

        out.write(output_image)
        cv2.imshow('MediaPipe Selfie Segmentation', output_image)
        if cv2.waitKey(5) & 0xFF == 27:
          break

    cap.release()
    out.release()
    end = time.time()
    return end - start

video_selfi_segmentation('./LJM/', 'cv_camera_sensor_stream_handler', './black')