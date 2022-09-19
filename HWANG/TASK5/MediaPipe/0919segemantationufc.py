import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# BG_COLOR = (192, 192, 192) # gray
BG_COLOR = (244, 244, 244) # white
# BG_COLOR = (0, 0, 0) # BLACK

file_name = './LJM/ufc_crop.mp4'

cap = cv2.VideoCapture(file_name)

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D', 'I', 'V', 'X'
delay = round(1000 / fps)

out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

# MODEL_SELECTION
# An integer index 0 or 1.
# Use 0 to select the general model, and 1 to select the landscape model (see details in Models).
# Default to 0 if not specified.
# 0이 조금 더 정확하게 나오는 경향이 있음.(노션에 두 모델 차이 정리해 둠.)
with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation:
  bg_image = None
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Can't find the input.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False # 무슨 의미가 있나_1?
    results = selfie_segmentation.process(image)

    image.flags.writeable = True # 무슨 의미가 있나_2?
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".

    # 값을 늘리면 좀 더 타이트하게 잡는 듯?
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.1

    # The background can be customized.
    #   a) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   b) Blur the input image by applying image filtering, e.g.,
    #      bg_image = cv2.GaussianBlur(image,(55,55),0)
    if bg_image is None:
      bg_image = np.zeros(image.shape, dtype=np.uint8)
      bg_image[:] = BG_COLOR
    output_image = np.where(condition, image, bg_image)
    out.write(output_image)
    cv2.imshow('MediaPipe Selfie Segmentation', output_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
out.release()
