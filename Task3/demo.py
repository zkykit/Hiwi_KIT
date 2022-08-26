
#from ast import arg
#import imghdr
#from re import I
import sys

#from cv2 import IMWRITE_PNG_STRATEGY_FILTERED, split
sys.path.append('core')# add path to the sys'envs variable

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

global i 
i = 0
DEVICE = 'cuda'

def load_image(imfile):# to have the tensor array 
    img = np.array(Image.open(imfile)).astype(np.uint8)#specically used for save image.rangefrom0to 255
    img = torch.from_numpy(img).permute(2, 0, 1).float()#transform tensor matrix
    return img[None].to(DEVICE)



#foto_base_path = '/media/zky/T7/task3/RAFT/frame/L00_frame'
#foto_names = os.listdir(foto_base_path)#to get foldername of pictures

def viz(img, flo,path_base_img):
    #print("##############################")
    img = img[0].permute(1,2,0).cpu().numpy()#[] inside isn't amount of pictures,no incluence nomatterhow
    flo = flo[0].permute(1,2,0).cpu().numpy()#for first object change tensor dims.
  
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)#picture stitching :pinjie


    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    global i
    #for foto_name in foto_names:

    #foto_name = '2022_03_10_S01_SC00_T01_SE002_L01'

    #foto_save_name = '/media/zky/T7/task3/RAFT/demo_file_save/L01_frame/'+f'{path_base_img}'+'/' 
    
    #print(path_base_img)
    
    #print(path_base_img)
    #create folder for pics to save
    #print("begin to create folder")
    #os.makedirs(foto_save_name,exist_ok=True)
    #cv2.imwrite('/media/zky/T7/task3/RAFT/demo_file_save/L01_frame/'+f'{path_base_img}'+'/'+f'{i}.png',img_flo[:, :, ::-1]) #


    '''
    saved_foto_path = '/media/zky/T7/task3/RAFT/frame_small/L00_frame/2022_03_10_S03_SC01_T02_SE002_L00'
    saved_names_temp = os.listdir(saved_foto_path)#now we have all pics name as list
    saved_names_imgs = saved_names_temp[0:]#from the first pic
    saved_names_imgs = sorted(saved_names_imgs)
    
    for saved_names_img in saved_names_imgs:
        saved_foto_name_base = saved_names_img.split('.')[0]#to get picture's number
        cv2.imwrite('/media/zky/T7/task3/RAFT/demo_file_save/L00_frame/'+foto_name+'/'+f'{saved_foto_name_base}.png',img_flo[:, :, ::-1]) #want to change i the same as name fotoname but not sequential

    '''
    cv2.imwrite('/media/zky/T7/task3/RAFT/L01_flow'+'/'+f'{i}.png',img_flo[:, :, ::-1]) #
    

def demo(args,path_base_img=None):
    model = torch.nn.DataParallel(RAFT(args))# use multiply gpu
    model.load_state_dict(torch.load(args.model))#used for read parameters
    model = model.module#diedai bianli all son layer of model
    model.to(DEVICE)
    model.eval()#guarantee all batch normalization use all varianz and middle value from trainingdata


    with torch.no_grad():#used for not calculate gradient
        images = glob.glob(os.path.join("/media/zky/T7/task3/RAFT/L01",'*.png'))#origin images path



        foto_save_name = '/media/zky/T7/task3/RAFT/L01_flow'#flow image saved path
        if not os.path.exists(foto_save_name):#create folder for L01
            os.makedirs(foto_save_name,exist_ok=True)
       
    

        for imfile1, imfile2 in zip(images[:-1], images[1:]):#delete last one, delete the first one
            
            #print("##########################")
            image1 = load_image(imfile1)#load from first image
            image2 = load_image(imfile2)#load from second image, meaning???
            #print(image1)
            

            padder = InputPadder(image1.shape)#class 'utils.utils.InputPadder
        
            image1, image2 = padder.pad(image1, image2)

            #print("#################")

        
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            #print(flow_up)
            #print(flow_low,flow_up)#cuda didnot be used,why???
            #print(flow_low) up and down of a single tensor array[]
            #print(flow_up)
            global i
            i=i+1

            #print("##############")
            viz(image1, flow_up,path_base_img)#no difference between image1 and image2 here
            #print("##########")
        #print(image1)#output 6 tensor 
            #print(image2)#separate output 9 tensor
            #print(type(imfile1)) #output image path 
        #print((imfile1))# 98.png be used, but why??? maybe because of InputPadder,it cause specific bianli, and choose the last
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    #parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    #here change
    

    path_base = '/media/zky/T7/task3/RAFT/frame_small/L01_frame'
    path_base_foldername = os.listdir(path_base)#get foldername as list
    path_base_imgs = path_base_foldername[0:]
    path_base_imgs = sorted(path_base_imgs)#path base imgs is foldername list

    demo(args)

    for path_base_img in path_base_imgs:
        #foto_save_name = '/media/zky/T7/task3/RAFT/demo_file_save/L01_frame/'+path_base_img+'/'
        #folder=os.makedirs(foto_save_name,exist_ok=True)#create folder for pics to save\\
        print(path_base_img)

        demo(args,path_base_img)
        #cv2.imwrite('/media/zky/T7/task3/RAFT/demo_file_save/L01_frame/'+f'{path_base_img}'+'/'+f'{i}.png',img_flo[:, :, ::-1]) #want to change i the same as name fotoname but not sequential
    #print("finish")  


