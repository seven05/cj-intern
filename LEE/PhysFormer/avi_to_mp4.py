# avi to mp4

import numpy as np
import cv2

#cap = cv2.VideoCapture('../data/lgi-ppgi-db/id3/cpi/cpi_talk/cv_camera_sensor_stream_handler.avi')
cap = cv2.VideoCapture('../lgi-ppgi-db/vid.avi')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('../data/lgi-ppgi-db/id3/cpi/cpi_talk/cv_camera_sensor_stream_handler.mp4',fourcc, 30.0, (640,480))
out = cv2.VideoWriter('../lgi-ppgi-db/vid.mp4',fourcc, 30.0, (640,480))
print("START")

print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # frame = cv2.flip(frame,0)
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print("THE END")
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()