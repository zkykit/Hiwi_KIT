"""

usecase :

author : 

data : 


"""
import argparse
from cProfile import label
import fnmatch
import glob
import json
import os
import shutil
import subprocess
from unicodedata import name
import uuid
import sys
from joblib import delayed
from joblib import Parallel
import pandas as pd


def create_url(url:str) ->str:#creat video url

    url_be = "https://cvhci.anthropomatik.kit.edu/~dschneider/jubot_demo_large/videos/"

    url = url_be + url.split("/")[-1]

    return url


def clip_video(video_url,start_time,end_time,output_file):#cut and save

    video_url = create_url(video_url)

    status = False
    command = ['ffmpeg',
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-i', "'%s'" % video_url,
               '-c:v', 'copy', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_file]
    command = ' '.join(command)#表示command之间有一个空格
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output
    
    status = os.path.exists(output_file)
    
    return status, 'Downloaded'


def create_save_folder(label_range,output_dir) ->dict:#creat save positon


    """

    usecase : create the classifer folder 

    return the cooresping path dict
    
    """

    label_to_dir = dict()#creat empty dict
    if not os.path.exists(output_dir):

        os.mkdir(output_dir)
    
    for i in range(label_range):

        this_dir = os.path.join(output_dir,"L"+str(i).zfill(2))#creat name for labelpart

        if not os.path.exists(this_dir):

            os.mkdir(this_dir)

        label_to_dir[f"{i}".zfill(2)] = this_dir

    return label_to_dir

        
def create_video_name(name_be:str,label:str,index:int,label_to_dir:list):#改名

    """
    
    
    """
    #creat name formal as "2022_03_10_S08_SC02_T02_SE001_L01.mp4"
    basename = name_be + "_SE" + str(index).zfill(3) + "_L" + label.zfill(2) + ".mp4"

    #print(basename)

    output_file = os.path.join(label_to_dir[label.zfill(2)],basename)

    return output_file



    pass#prevent situations that not allow empty context.


def read_annotation(input_json):#load json

    """
    usecase : read the file to get the correspding annotation

    input_json : the path of the json file

    return a list 
    
    """

    with open(input_json,"r") as f:

        data= json.load(f)

    return data


def name_beg(name) ->str:#这段给check_status用的

    basename = name.split('/')[-1]

    usename=basename.split(".")[0]

    return usename


def check_status(video_url:str,tricks:list,label_to_dir:dict) -> list:

    status_list = []

    for index,seg in enumerate(tricks,start = 1):
        
        ######################
        ##### step 1 create the fast name 
        name_be = name_beg(video_url)#read useful-part in url

        start_time = seg["start"]

        end_time = seg["end"]

        label = seg["labels"][0]

        output_file = create_video_name(name_be,label,index,label_to_dir)

        clip_id = os.path.basename(output_file).split('.mp4')[0]

        if os.path.exists(output_file):
            
            status = tuple([clip_id, True, 'Exists'])

            status_list.append(status)
            
            continue
        else:
            downloaded, log=clip_video(video_url,start_time,end_time,output_file)

            status = tuple([clip_id, downloaded, log])

            status_list.append(status)
        
    return status_list


def main(
    input_json,
    output_dir=None,
    label_range=10
    ):

    annotations = read_annotation(input_json)

    label_to_dir = create_save_folder(label_range,output_dir)

    for anno in annotations:

        
        status_lst = check_status(anno["video_url"],anno["tricks"],label_to_dir)

    with open('download_report.json', 'w') as fobj:
            
        fobj.write(json.dumps(status_lst))


if __name__ == "__main__":

    main(
        input_json="/Users/kyz/Desktop/hiwi_task/task2/jubot_demo_corrected_annotations.json",

        output_dir="/Users/kyz/Desktop/hiwi_task/video1",

        label_range=11

        )
