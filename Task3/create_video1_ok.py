import cv2
import numpy as np
import glob
import skvideo.io
import os 
def create_video():
    
#path for save
    out = cv2.VideoWriter('/media/zky/T7/task3/RAFT/video_save/project_L01_flow.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, (960,1088))
    #path of fotos
    for filename in glob.glob('/media/zky/T7/task3/RAFT/demo_file_save/test/*.png'):

        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        out.write(img)
    out.release()

create_video()