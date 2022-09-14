import cv2
import os
import numpy as np
'''
try:
    if not os.path.exists('./Crop'):
        os.makedirs('./Crop')
except OSError:
    print('error : creating dir Crop')

for x in range(0,7000):
    count = 1
    img = cv2.imread("Captures/{}.jpg".format(x), 0)
    crop_1 = img[5:25,4:15]
    cv2.imwrite('./Crop' + "/%d_%d.jpg" %(x,count), crop_1)
    count += 1
    crop_2 = img[5:25,15:27]
    cv2.imwrite('./Crop' + "/%d_%d.jpg" %(x,count), crop_2)
    count += 1
    crop_3 = img[5:25,34:45]
    cv2.imwrite('./Crop' + "/%d_%d.jpg" %(x,count), crop_3)
    count += 1
    crop_4 = img[5:25,45:56]
    cv2.imwrite('./Crop' + "/%d_%d.jpg" %(x,count), crop_4)
    count = 0
'''
try:
    if not os.path.exists('./Crop'):
        os.makedirs('./Crop')
except OSError:
    print('error : creating dir Crop2')

for y in range(132,150):
    count = 1
    img = cv2.imread("Captures2/{}.jpg".format(y), 0)
    crop_1 = img[11:33,5:22]
    cv2.imwrite('./Crop' + "/%d_%d.jpg" %(y,count), crop_1)
    count += 1
    crop_2 = img[11:33,23:40]
    cv2.imwrite('./Crop' + "/%d_%d.jpg" %(y,count), crop_2)
    count += 1
    crop_3 = img[11:33,50:67] 
    cv2.imwrite('./Crop' + "/%d_%d.jpg" %(y,count), crop_3)
    count += 1
    crop_4 = img[11:33,68:85]
    cv2.imwrite('./Crop' + "/%d_%d.jpg" %(y,count), crop_4)
    count = 0

'''
img2 = cv2.imread("Crop/200_1.jpg")
h, w, c = img2.shape
'''
'''
print('width:  ', w)
print('height: ', h)
print('channel:', c)
'''
'''
np.array
for i in h:
    for j in w:
'''     

exit()