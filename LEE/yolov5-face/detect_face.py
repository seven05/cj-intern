# -*- coding: UTF-8 -*-
import argparse
from configparser import Interpolation
from locale import normalize
import time
from pathlib import Path
from turtle import down

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import copy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    label = str(f"{x2-x1}, {y2-y1}")
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def blacken(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = max(int(xyxy[0])-50, 0)
    y1 = max(int(xyxy[1])-50, 0)
    x2 = min(int(xyxy[2])+50, w)
    y2 = min(int(xyxy[3])+50, h)
    
    # cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
    crop_img = img[y1:y2, x1:x2]

    ###### 1. 센터로 안 몰고 박스 외 부분은 blacken
    # img1 = np.pad(crop_img[:,:,0], ((y1, h-y2), (x1, w-x2)), 'constant', constant_values=0)
    # img2 = np.pad(crop_img[:,:,1], ((y1, h-y2), (x1, w-x2)), 'constant', constant_values=0)
    # img3 = np.pad(crop_img[:,:,2], ((y1, h-y2), (x1, w-x2)), 'constant', constant_values=0)
    
    ###### 2. 센터로 몰고 박스 외 부분도 blacken
    ylen = int((y2-y1)/2)
    xlen = int((x2-x1)/2)
    yup = int((h-(y2-y1))/2)
    ydown = int(h-yup-(y2-y1))
    xup = int((w-(x2-x1))/2)
    xdown = int(w-xup-(x2-x1))

    img1 = np.pad(crop_img[:,:,0], ((yup, ydown), (xup, xdown)), 'constant', constant_values=0)
    img2 = np.pad(crop_img[:,:,1], ((yup, ydown), (xup, xdown)), 'constant', constant_values=0)
    img3 = np.pad(crop_img[:,:,2], ((yup, ydown), (xup, xdown)), 'constant', constant_values=0)

    retimg = np.dstack((img1, img2, img3))
    print(retimg.shape)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    # 눈코입 표시 현재는 필요 x
    # for i in range(5):
    #     point_x = int(landmarks[2 * i])
    #     point_y = int(landmarks[2 * i + 1])
    #     cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    return retimg



def detect_one(model, image_path, device):
    # Load model
    img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xyxy, conf, landmarks, class_num)

    cv2.imwrite('result.jpg', orgimg)


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    #     model = TracedModel(model, device, opt.img_size)

    # if half:
    #     model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


    seg_weights = FCN_ResNet50_Weights.DEFAULT
    seg_model = fcn_resnet50(wieghts=seg_weights)
    seg_model.eval()
    preprocess = seg_weights.transforms()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        # view_img = check_imshow()
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        pass
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    bxyxy = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression_face(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        h = img.shape[0]
        w = img.shape[1]
        # Process detections

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                argmax = 0
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    #conf = det[j, 4].cpu().numpy()
                    #landmarks = det[j, 5:15].view(-1).tolist()
                    #class_num = det[j, 15].cpu().numpy()
                    #im0 = show_results(im0, xyxy, conf, landmarks, class_num)

                    b = det[argmax, :4].view(-1).tolist()
                    # 가장 큰 box가 선수
                    # 이후 이 부분은 classification으로 바뀌어야 함.!
                    if b[2]-b[0]<xyxy[2]-xyxy[0] and b[3]-b[1]<xyxy[3]-xyxy[1]:
                        argmax = j
                
                plt.imshow(im0)
                plt.show()
                bxyxy = det[argmax, :4].view(-1).tolist()
                conf = det[argmax, 4].cpu().numpy()
                landmarks = det[argmax, 5:15].view(-1).tolist()
                class_num = det[argmax, 15].cpu().numpy()
                #im0 = show_results(im0, bxyxy, conf, landmarks, class_num)
                im0 = blacken(im0, bxyxy, conf, landmarks, class_num)
                

            else:
                # 안잡혔으면 이전 box 재탕
                if len(bxyxy): # 맨 첫번째 프레임은 그리지마 (못그려)
                    im0 = blacken(im0, bxyxy, conf, landmarks, class_num)
                    

            pil_im0 = Image.fromarray(im0)
            batch = preprocess(pil_im0).unsqueeze(0)
            seg_pred = seg_model(batch)["out"]
            normalized_masks = seg_pred.softmax(dim=1)
            class_to_idx = {cls: idx for (idx, cls) in enumerate(seg_weights.meta["categories"])}
            mask = normalized_masks[0, class_to_idx["person"]]
            mask = np.where(mask>=0.5, 0, 255)
            mask = np.array(mask, dtype='uint8')
            mask = cv2.resize(mask, dsize=(im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask = np.expand_dims(mask, axis=2)
            # 곱해서 배경 날리도록
            im0 = im0*mask

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s-face.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../lgi-ppgi-db/ufc_crop.mp4', help='source') # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    #detect_one(model, opt.image, device)
    detect()
