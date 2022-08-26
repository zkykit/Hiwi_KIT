from tkinter import Frame
from turtle import shape
from typing import List
import cv2
import os
import torch
import torchvision.transforms as transforms

import sys

#from cv2 import IMWRITE_PNG_STRATEGY_FILTERED, split
sys.path.append('core')# add path to the sys'envs variable

import argparse
import glob
import numpy as np
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

class create_flow():


    def __init__(self,args) -> None:


        self.path = args.path

        self.model = args.model

        self.resize = args.resize

        self.save = args.save

        self.video_save_path = None

        self.main()

    
    def photo_resize(self,frame,resize_rate=0.5):
      
        w = int(frame.shape[1] * resize_rate)
        h = int(frame.shape[0] * resize_rate)
        size = (w,h)
        frame = cv2.resize(frame,size)

        return frame

    @staticmethod
    def cv2_to_torch(frame):

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = np.array(frame).astype(np.uint8)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()

        return frame[None].to(DEVICE)

    def run(self,video):

        video_name = video.split(".")[0]

        try:

            cap = cv2.VideoCapture(self.video) #  path need to be change
            out = cv2.VideoWriter(os.path.join(self.video_save_path,video_name+".mp4"),cv2.VideoWriter_fourcc(*"mp4v"), 30, (960,1088))

            if not cap.isOpened():

                print(f"the video {video} can not be opened, please check")

                return 0
            
            else:

                ret,frame_1 = cap.read()
                
                count = True

                while ret:

                    ret,frame_2 = cap.read()

                    if not ret:

                        break

                    if self.resize:

                        if count:

                            frame_1 = self.photo_resize(frame_1)
                            frame_2 = self.photo_resize(frame_2)
                        else:

                            frame_2 = self.photo_resize(frame_2)
                    if count:
                        frame_1 = self.cv2_to_torch(frame_1)
                        frame_2 = self.cv2_to_torch(frame_2)
                    else:
                        frame_2 = self.cv2_to_torch(frame_2)
                    
                    
                    ############################
                    #### use the demo script

                    img_flo=self.demo(frame_1,frame_2)
                    cv2.imwrite("./test.png",img_flo)
                    out.write(img_flo)

                    ###########################
                    #### update frame 

                    frame_1 = frame_2
                    count = False
                
        finally:
            cap.release()
            out.release()
            print("finish one")

    @staticmethod
    def vis(img, flo):

        img = img[0].permute(1,2,0).cpu().numpy()#[] inside isn't amount of pictures,no incluence nomatterhow
        flo = flo[0].permute(1,2,0).cpu().numpy()#for first object change tensor dims.
    
        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)#picture stitching :pinjie
        img_flo = np.array(img_flo).astype(np.uint8)
        img_flo=cv2.cvtColor(img_flo,cv2.COLOR_RGB2BGR)

        return img_flo
        

    def demo(self,image1,image2):
        model = torch.nn.DataParallel(RAFT(args))# use multiply gpu
        model.load_state_dict(torch.load(args.model))#used for read parameters
        model = model.module#diedai bianli all son layer of model
        model.to(DEVICE)
        model.eval()#guarantee all batch normalization use all varianz and middle value from trainingdata

        with torch.no_grad():#used for not calculate gradient
        
            padder = InputPadder(image1.shape)#class 'utils.utils.InputPadder
        
            image1, image2 = padder.pad(image1, image2)
        
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            img_flo = self.vis(image1, flow_up)#no difference between image1 and image2 here
            return img_flo

    def main(self):

        video_paths = os.listdir(self.path)

        for vp in video_paths:

            video_save_path = vp

            self.video_save_path = os.path.join(self.save,video_save_path)

            if not os.path.exists(self.video_save_path):
                
                os.makedirs(self.video_save_path)

            self.video_path = os.path.join(self.path,vp)

            videos = os.listdir(self.video_path)

            for video in videos:

                self.video = os.path.join(self.video_path,video)

                self.run(video)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--resize', action='store_true', help='if need to resize the photo')
    parser.add_argument('--save', help='give the save path')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    create_flow(args)
        