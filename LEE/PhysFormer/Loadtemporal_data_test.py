from __future__ import print_function, division
from email.mime import image
import os
from sre_constants import SUCCESS
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import pdb
import math

from skimage.util import img_as_float


clip_frames = 160+60   
#clip_frames = 100+60

class Normaliztion (object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        video_x, framerate, clip_average_HR_peaks = sample['video_x'],sample['framerate'],sample['clip_average_HR_peaks']
        new_video_x = (video_x - 127.5)/128
        return {'video_x': new_video_x, 'framerate':framerate, 'clip_average_HR_peaks':clip_average_HR_peaks}




class ToTensor (object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video_x, framerate, clip_average_HR_peaks = sample['video_x'],sample['framerate'],sample['clip_average_HR_peaks']

        # swap color axis because
        # numpy image: clip x depth x H x W x C
        # torch image: clip x C x depth X H X W
        video_x = video_x.transpose((0, 4, 1, 2, 3))
        video_x = np.array(video_x)
        
        framerate = np.array(framerate)
        
        clip_average_HR_peaks = np.array(clip_average_HR_peaks)
        
        return {'video_x': torch.from_numpy(video_x.astype(np.float)).float(), 'framerate': torch.from_numpy(framerate.astype(np.float)).double(), 'clip_average_HR_peaks': torch.from_numpy(clip_average_HR_peaks.astype(np.float)).float()}

class VIPL (Dataset):
    
    def __init__(self, info_list, root_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        video_path = os.path.join(self.root_dir, str(self.landmarks_frame.iloc[idx, 0]))
        
        total_clips = self.landmarks_frame.iloc[idx, 1]
        
        video_x = self.get_single_video_x(video_path, total_clips)
        
        framerate  = self.landmarks_frame.iloc[idx, 2]
        
        clip_average_HR  = self.landmarks_frame.iloc[idx, 3]
		    
        
        sample = {'video_x': video_x, 'framerate':framerate, 'clip_average_HR_peaks':clip_average_HR}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_video_x(self, video_path, total_clips):
        video_jpgs_path = video_path

        video_x = np.zeros((total_clips, clip_frames, 128, 128, 3))
        
        for tt in range(total_clips):
            image_id = tt*160 + 61
            #image_id = tt*100 + 61
            for i in range(clip_frames):
                #s = "%05d" % image_id
                s = str(image_id)
                #image_name = 'image_' + s + '.png'
                image_name = 'frame_' + s + '.jpg'
    
                # face video 
                image_path = os.path.join(video_jpgs_path, image_name)
                print(image_path)

                # if image_id>=392:
                #     break
                
                tmp_image = cv2.imread(image_path)
                
                #if tmp_image is None:    # It seems some frames missing 
                #    tmp_image = cv2.imread(self.root_dir+'p30/v1/source2/image_00737.png')
                    
                tmp_image = cv2.resize(tmp_image, (132, 132), interpolation=cv2.INTER_CUBIC)[2:130, 2:130, :]
                
                video_x[tt, i, :, :, :] = tmp_image  
                            
                image_id += 1
   
        return video_x

# info_list VIPL_fold1_test1.txt
# root_dir ./
class MyDataset(Dataset):
    def __init__(self, info_list, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

        video_path = os.path.join(self.root_dir, str(self.landmarks_frame.iloc[0, 0]))
        print("here!!!!!")
        #video = torchvision.io.read_video("../data/ubfc_phys/s1/s1/vid_s1_T1.avi")
        #print(video)
        video_path = "../data/ubfc_phys/s1/s1/vid_s1_T1.avi"
        video = cv2.VideoCapture(video_path)

        sucess = True
        success, img = video.read()

        totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_torch = np.zeros((totalFrames, 128, 128, 3), dtype = np.float32)
        print(totalFrames)
        
        i=0

        # 여기 코드 수정해서!!!!! yolov5로 crop 후 해봐라 꼭
        while sucess and i<6325:
            new_img = img_as_float(img)
            vidLxL = cv2.resize(new_img, (128, 128), interpolation = cv2.INTER_AREA)
            vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
            vidLxL[vidLxL > 1] = 1
            vidLxL[vidLxL < (1/255)] = 1/255

            #print("vid:", vidLxL.shape)
            video_torch[i, :, :, :] = vidLxL
            if i%10==0:
                print(i)
                print(vidLxL.shape)

            success, img = video.read()
            i+=1

        print("original shape:", video_torch.size)
        video_torch = np.transpose(video_torch, (0, 3, 1, 2))
        print("later:", video_torch.size)
        self.video = video_torch

    def __len__(self):
        return len(self.landmarks_frame)
   
    def __getitem__(self, idx):
        total_clips = self.landmarks_frame.iloc[idx, 1]

        print(self.video.size)
        print("got here!!!!!")
        video_x = self.video[idx, :, :, :]
        
        framerate  = self.landmarks_frame.iloc[idx, 2]
        
        clip_average_HR  = self.landmarks_frame.iloc[idx, 3]
        
        sample = {'video_x': video_x, 'framerate':framerate, 'clip_average_HR_peaks':clip_average_HR}

        if self.transform:
            sample = self.transform(sample)
        return sample




 


