import cv2
import matplotlib.pyplot as plt
import os
vidcap = cv2.VideoCapture("../lgi-ppgi-db/ufc_crop.mp4")
success, img = vidcap.read()

framecount = 0


save_folder = "./for_phys_ufc_crop"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    os.mkdir(save_folder+'/out')


while success:
    # plt.imshow(img)
    # plt.show()
    print(framecount)
    h, w, c = img.shape
    nh = int(h/5)
    nw = int(w/5)
    
    crop_img = img[1*nh:4*nh,1*nw:4*nw]
    crop_img = img
    
    # if framecount%10==0:
    #     plt.imshow(crop_img)
    #     plt.show()
    #     break

    #print(save_folder+f"/frame_{framecount}.jpg")
    cv2.imwrite(save_folder+f"/frame_{framecount}.jpg", crop_img)

    sucess, img = vidcap.read()
    framecount+=1

print("done")