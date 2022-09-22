import time
start = time.time()
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# BG_COLOR = (192, 192, 192) # gray
BG_COLOR = (244, 244, 244) # white
# BG_COLOR = (0, 0, 0) # BLACK

file_directory = './LJM/'
file_name = 'ufc_crop_yolo'
# outfile_directory = ''

outfile_name = 'output.avi'
file_name = file_directory + file_name +'.mp4'


cap = cv2.VideoCapture(file_name)

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D', 'I', 'V', 'X'
# delay = round(1000 / fps)

out = cv2.VideoWriter(outfile_name, fourcc, fps, (w, h))

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
      # print("Can't find the input.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False # 무슨 의미가 있나_1? => 내가 받는 output 영상에 영향을 주는 변수가 아니라 reference로 사용가능한지 그걸 하는 듯.
    results = selfie_segmentation.process(image)

    image.flags.writeable = True # 무슨 의미가 있나_2?
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    # 값을 늘리면 좀 더 타이트하게 잡는 듯?
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.8

    # 변경해가면서 진행하는 것(axis = -1 혹은 2로 해야함 or 에러 발생) * 다음의 숫자는 3이 아니면 에러가 발생 <= 아마 크기가 맞지 않는 듯
    # 뒤의 숫자만 내가 변경 가능한 부분일 듯(아마도 blur하는 부분에서 진행되는 변수의 차이라 생각)
    # condition = np.stack(
    #   (results.segmentation_mask,) * 3, axis=-1) > 0.0000001

    # The background can be customized.
    #   a) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   b) Blur the input image by applying image filtering, e.g.,
    #      bg_image = cv2.GaussianBlur(image,(55,55),0)
    # bg_image = cv2.GaussianBlur(image, (55, 55), 0)
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
end = time.time()

print("running time is ", end - start)