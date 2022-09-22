import cv2
import mediapipe as mp
import math

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

        annotated_image = image.copy()
        size = -1
        for detection in results.detections:
            location_data = detection.location_data
            bb = location_data.relative_bounding_box

            # bb_box = [bb.xmin, bb.ymin, bb.width, bb.height]
            # print(f"RBBox: {bb_box}")
            #
            # print('Nose tip:')
            # print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            # mp_drawing.draw_detection(annotated_image, detection)
            if size < bb.width * bb.height:
                x_min, y_min, width, height = bb.xmin, bb.ymin, bb.width, bb.height
                size = bb.width * bb.height

            # pt1 = math.floor(x_min * W), math.floor(y_min * H)
            # pt2 = math.floor(x_min * W) + math.ceil(width * W), math.floor(y_min * H) + math.ceil(height * H)
            # yellow = (0, 255, 255)
            # cv2.rectangle(annotated_image, pt1, pt2, yellow)

        # cv2.imwrite('results.png', annotated_image)
        #     cv2.imshow('result', annotated_image)
        # cv2.waitKey(0)
    return max(0, math.floor(x_min * W)), max(0, math.floor(y_min * H)), math.ceil(height * H), math.ceil(width * W)

