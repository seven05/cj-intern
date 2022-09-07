import cv2
import matplotlib.pyplot as plt
vidcap = cv2.VideoCapture("../../data/lgi-ppgi-db/id3/cpi/cpi_gym/cv_camera_sensor_stream_handler.mp4")
success, img = vidcap.read()

framecount = 0
while success:
    # plt.imshow(img)
    # plt.show()
    print(framecount)
    h, w, c = img.shape
    nh = int(h/3)
    nw = int(w/4)
    crop_img = img[:,1*nw:3*nw]

    # plt.imshow(crop_img)
    # plt.show()
    cv2.imwrite(f"./for_physformer2/cpi_{framecount}.jpg", crop_img)

    sucess, img = vidcap.read()
    framecount+=1

print("done")