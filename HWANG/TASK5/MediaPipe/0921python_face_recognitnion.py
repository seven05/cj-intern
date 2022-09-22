import face_recognition
from matplotlib import pyplot as plt
import cv2
from PIL import Image
# 실패
# open할 이미지 경로
imgfile = './face_recognition/test_input/1.jpg'
savepath = './face_recognition/test_output/'
file = '1.jpg'
image = face_recognition.load_image_file(imgfile)
face_locations = face_recognition.face_locations(image)
# 눈코입 찾아서 얼굴있으면 개수 알려줌
print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:
    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print(
        "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    p_image = pil_image.resize((48, 48))

    # pil_image.show()
    plt.imshow(p_image)
    p_image.save(savepath + 'f_' + file)